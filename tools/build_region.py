#!/usr/bin/env python3
"""
Turns an OpenStreetMap extract (already reduced by `osmium tags-filter` and exported with
`osmium export -f geojsonseq`) into a compact PriusMFD map file (.pmd).

Only standard library. Reads GeoJSON features (one per line, RS-separated or not) from stdin.

File layout (little endian):
  header (64 bytes)
    0  4s  magic b"PMD1"
    4  u16 version (1)
    6  u16 reserved
    8  i32 cell size in 1e-6 degrees (10000 = 0.01°)
    12 i32 first cell row (lat / cell size)
    16 i32 first cell column (lon / cell size)
    20 i32 rows
    24 i32 cols
    28 i64 build time (unix seconds)
    36 u32 road index offset   – (rows*cols+1) u32, offsets into road data
    40 u32 road data offset
    44 u32 poi index offset    – (rows*cols+1) u32, offsets into poi data
    48 u32 poi data offset
  road chunk: u8 speed forward, u8 speed backward, u8 flags (bit0 = mph), u8 n,
              n × (i16 dlat, i16 dlon) relative to the cell's south-west corner, 1e-6 degrees
  poi:        u8 kind, i16 dlat, i16 dlon
Speeds: 0 unknown, 1–250 value, 254 living street / walking pace, 255 no limit.
"""
import json
import math
import re
import struct
import sys
import time
from collections import defaultdict

CELL = 10_000            # 0.01 degree in 1e-6 units
MAX_SEG_M = 800.0        # longer segments are split so every point fits an int16 offset
UNKNOWN, LIVING, NONE = 0, 254, 255

# POI kinds (must match the app)
LEVEL_CROSSING, ZEBRA, ROUNDABOUT, CHILDREN, ANIMALS, CURVE, DANGER, STOP = 1, 2, 3, 4, 5, 6, 7, 8

RAIL = {"rail", "narrow_gauge", "light_rail", "preserved"}

# Implicit limits per country code (km/h; mph flagged separately)
IMPLICIT = {
    "urban": 50, "rural": 100, "motorway": NONE, "living_street": LIVING, "walk": LIVING,
    "bicycle_road": 30, "school_zone": 30,
}
COUNTRY = {
    "AT": {"rural": 100, "motorway": 130, "trunk": 100},
    "CH": {"rural": 80, "motorway": 120, "trunk": 100},
    "NL": {"rural": 80, "motorway": 100},
    "BE": {"rural": 70, "motorway": 120},
    "LU": {"rural": 90, "motorway": 130},
    "DK": {"rural": 80, "motorway": 130},
    "CZ": {"rural": 90, "motorway": 130},
    "PL": {"rural": 90, "motorway": 140, "expressway": 120},
    "FR": {"rural": 80, "motorway": 130},
    "IT": {"rural": 90, "motorway": 130},
}
GB_MPH = {"nsl_single": 60, "nsl_dual": 70, "motorway": 70, "urban": 30, "nsl_restricted": 30}


def parse_speed(v):
    """Returns (speed, mph) or None if the value says nothing usable."""
    if v is None:
        return None
    s = v.strip()
    low = s.lower()
    if low == "none":
        return NONE, False
    if low in ("walk", "living_street"):
        return LIVING, False
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(mph|km/h|kmh)?", low)
    if m:
        n = int(round(float(m.group(1))))
        if 1 <= n <= 250:
            return n, m.group(2) == "mph"
        return None
    m = re.fullmatch(r"([a-z]{2}):(.+)", low)
    if m:
        cc, kind = m.group(1).upper(), m.group(2)
        z = re.fullmatch(r"zone:?(\d+)", kind)
        if z:
            return int(z.group(1)), cc in ("GB", "US")
        if cc == "GB" and kind in GB_MPH:
            return GB_MPH[kind], True
        val = COUNTRY.get(cc, {}).get(kind, IMPLICIT.get(kind))
        if val is not None:
            return val, False
    return None


def way_speeds(tags):
    """(forward, backward, mph) for a road way."""
    def first(*keys):
        for k in keys:
            r = parse_speed(tags.get(k))
            if r:
                return r
        return None
    base = first("maxspeed", "maxspeed:type", "source:maxspeed", "zone:maxspeed", "zone:traffic")
    if base is None and tags.get("highway") == "living_street":
        base = (LIVING, False)
    fwd = parse_speed(tags.get("maxspeed:forward")) or base
    back = parse_speed(tags.get("maxspeed:backward")) or base
    f = fwd[0] if fwd else UNKNOWN
    b = back[0] if back else UNKNOWN
    mph = bool((fwd and fwd[1]) or (back and back[1]))
    return f, b, mph


def poi_kind(tags):
    if tags.get("railway") == "level_crossing":
        return LEVEL_CROSSING
    if tags.get("highway") == "stop":
        return STOP
    hz = tags.get("hazard", "")
    ts = tags.get("traffic_sign", "")
    if hz in ("children", "school_zone") or re.search(r"\bDE:136\b", ts):
        return CHILDREN
    if hz in ("animal_crossing", "deer", "cattle", "horse", "frog") or re.search(r"\bDE:14[2-4]\b", ts):
        return ANIMALS
    if hz.startswith("curve") or re.search(r"\bDE:10[35]", ts):
        return CURVE
    if hz or re.search(r"\bDE:101\b", ts):
        return DANGER
    if tags.get("highway") == "crossing" and (
        tags.get("crossing_ref") == "zebra" or tags.get("crossing") in ("zebra", "uncontrolled", "marked")
        or tags.get("crossing:markings") == "zebra"
    ):
        return ZEBRA
    return None


def e6(x):
    return int(round(x * 1_000_000))


class Builder:
    def __init__(self):
        self.roads = defaultdict(bytearray)
        self.pois = defaultdict(bytearray)
        self.round_seen = set()
        # level crossings only count on real railway lines (trams cross roads everywhere in cities)
        self.rail_nodes = set()
        self.pending_crossings = []
        self.n_roads = self.n_pois = 0

    @staticmethod
    def cell_of(lat6, lon6):
        return lat6 // CELL, lon6 // CELL

    def add_way(self, coords, tags):
        hw = tags.get("highway")
        if not hw:
            return
        if tags.get("junction") in ("roundabout", "circular"):
            self.add_roundabout(coords)
        fwd, back, mph = way_speeds(tags)
        oneway = tags.get("oneway")
        if oneway == "-1":
            coords = list(reversed(coords)); fwd, back = back, fwd
        if oneway in ("yes", "1", "true", "-1") or tags.get("junction") in ("roundabout", "circular") or hw == "motorway":
            back = fwd
        pts = [(e6(lat), e6(lon)) for lon, lat in coords]
        # split long segments
        dense = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            dlat = (b[0] - a[0]) * 0.1113
            dlon = (b[1] - a[1]) * 0.1113 * math.cos(math.radians(a[0] / 1e6))
            n = max(1, math.ceil(math.hypot(dlat, dlon) / MAX_SEG_M))
            for i in range(1, n + 1):
                dense.append((a[0] + (b[0] - a[0]) * i // n, a[1] + (b[1] - a[1]) * i // n))
        chunk_cell, chunk = None, []

        def flush():
            if len(chunk) >= 2:
                r, c = chunk_cell
                buf = self.roads[chunk_cell]
                buf += struct.pack("<BBBB", fwd, back, 1 if mph else 0, len(chunk))
                for la, lo in chunk:
                    buf += struct.pack("<hh", la - r * CELL, lo - c * CELL)
                self.n_roads += 1

        for a, b in zip(dense, dense[1:]):
            if a == b:
                continue
            cell = self.cell_of((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            if cell == chunk_cell and chunk and chunk[-1] == a and len(chunk) < 255:
                chunk.append(b)
            else:
                flush()
                chunk_cell, chunk = cell, [a, b]
        flush()

    def add_roundabout(self, coords):
        lat = sum(c[1] for c in coords) / len(coords)
        lon = sum(c[0] for c in coords) / len(coords)
        key = (round(lat * 3000), round(lon * 2000))  # ~35 m grid: pieces of one roundabout collapse
        if key in self.round_seen:
            return
        self.round_seen.add(key)
        self.add_poi(lat, lon, ROUNDABOUT)

    def add_poi(self, lat, lon, kind):
        la, lo = e6(lat), e6(lon)
        r, c = self.cell_of(la, lo)
        self.pois[(r, c)] += struct.pack("<Bhh", kind, la - r * CELL, lo - c * CELL)
        self.n_pois += 1

    def feature(self, f):
        g = f.get("geometry") or {}
        tags = f.get("properties") or {}
        t = g.get("type")
        if t == "LineString" and tags.get("railway") in RAIL and not tags.get("highway"):
            for lon, lat in g["coordinates"]:
                self.rail_nodes.add((e6(lat), e6(lon)))
        elif t == "LineString":
            if len(g["coordinates"]) >= 2:
                self.add_way(g["coordinates"], tags)
        elif t == "Point":
            k = poi_kind(tags)
            if k == LEVEL_CROSSING:
                lon, lat = g["coordinates"]
                self.pending_crossings.append((lat, lon))
            elif k:
                lon, lat = g["coordinates"]
                self.add_poi(lat, lon, k)

    def write(self, out):
        for lat, lon in self.pending_crossings:
            if (e6(lat), e6(lon)) in self.rail_nodes:
                self.add_poi(lat, lon, LEVEL_CROSSING)
        cells = set(self.roads) | set(self.pois)
        if not cells:
            raise SystemExit("no data")
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        rows, cols = r1 - r0 + 1, c1 - c0 + 1

        def section(store):
            idx = bytearray(); data = bytearray()
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    idx += struct.pack("<I", len(data))
                    data += store.get((r, c), b"")
            idx += struct.pack("<I", len(data))
            return idx, data

        ridx, rdata = section(self.roads)
        pidx, pdata = section(self.pois)
        o_ridx = 64
        o_rdata = o_ridx + len(ridx)
        o_pidx = o_rdata + len(rdata)
        o_pdata = o_pidx + len(pidx)
        header = struct.pack("<4sHHiiiiiqIIII", b"PMD1", 1, 0, CELL, r0, c0, rows, cols, int(time.time()),
                             o_ridx, o_rdata, o_pidx, o_pdata)
        header += b"\0" * (64 - len(header))
        out.write(header); out.write(ridx); out.write(rdata); out.write(pidx); out.write(pdata)
        return {
            "bbox": [r0 * CELL / 1e6, c0 * CELL / 1e6, (r1 + 1) * CELL / 1e6, (c1 + 1) * CELL / 1e6],
            "roads": self.n_roads, "pois": self.n_pois,
        }


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: osmium export ... | build_region.py out.pmd")
    b = Builder()
    for line in sys.stdin:
        line = line.strip().lstrip("\x1e")
        if line:
            b.feature(json.loads(line))
    with open(sys.argv[1], "wb") as out:
        meta = b.write(out)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
