"""
Stage 3 — net risk score for one place + window.

Reads (read-only): incidents, places, tracked_objects
Writes: net_risk_scores  (one row per place_id, window; re-run replaces it)

Run:
    python stage3_score.py --db schema.db --place P1 --source S1

Does two passes automatically: rain-off ("this_clip") and rain-on
("this_clip_rain"), and prints how much R moved between them.
"""
from __future__ import annotations

import argparse
import math
import sqlite3
import uuid
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Named constants — tune these on a real clip, then freeze them.
# ---------------------------------------------------------------------------

# MoRTH persons-killed shares (2024-style). Update S_K when a newer report
# is out. lane_cut has no MoRTH line -> small team-chosen placeholder.
S_K: Dict[str, float] = {
    "speeding": 70.0,
    "wrong_way": 5.5,
    "red_light": 0.3,
    "lane_cut": 2.0,
}

# Place-card multipliers (frozen). city_prior comes from the places table.
C_RAIN = 1.15
C_NIGHT = 1.10
C_RUSH = 1.10
C_HIGHWAY = 1.10

# Score-combination weights: R = A_V*V + B_P*P + C_E*E + D_C*C
A_V = 20.0
B_P = 15.0
C_E = 3.0
D_C = 10.0

BAND_COLD_MAX = 39
BAND_WARM_MAX = 69

VIDEO_TYPES = ("speeding", "wrong_way", "red_light", "lane_cut")
INFERRED_TYPES = ("near_miss", "harsh_brake")


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------

def compute_weights(s_k: Dict[str, float]) -> Dict[str, float]:
    """Log-normalise MoRTH shares so rare-but-serious types don't vanish."""
    logs = {k: math.log(1 + v) for k, v in s_k.items()}
    total = sum(logs.values())
    return {k: v / total for k, v in logs.items()}


def band_for(R: float) -> str:
    if R <= BAND_COLD_MAX:
        return "cold"
    if R <= BAND_WARM_MAX:
        return "warm"
    return "hot"


# ---------------------------------------------------------------------------
# DB reads (all bound params, never f-strung)
# ---------------------------------------------------------------------------

def compute_T(conn: sqlite3.Connection, place_id: str, source_id: str) -> float:
    row = conn.execute(
        "SELECT MIN(ts_ms), MAX(ts_ms) FROM incidents WHERE place_id=? AND source_id=?",
        (place_id, source_id),
    ).fetchone()
    if row is None or row[0] is None:
        return 1 / 60
    duration_min = (row[1] - row[0]) / 60000
    return max(duration_min, 1 / 60)


def count_types(
    conn: sqlite3.Connection, place_id: str, source_id: str, source: str, wanted: tuple
) -> Dict[str, int]:
    rows = conn.execute(
        "SELECT type, COUNT(*) FROM incidents "
        "WHERE place_id=? AND source_id=? AND source=? GROUP BY type",
        (place_id, source_id, source),
    ).fetchall()
    counts = {t: 0 for t in wanted}
    for t, n in rows:
        if t in counts:
            counts[t] = n
    return counts


def count_exposure(conn: sqlite3.Connection, place_id: str, source_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(DISTINCT track_id) FROM tracked_objects "
        "WHERE place_id=? AND source_id=?",
        (place_id, source_id),
    ).fetchone()
    return row[0] if row and row[0] else 0


def load_place(conn: sqlite3.Connection, place_id: str) -> Dict[str, object]:
    row = conn.execute(
        "SELECT road_kind, is_rain, is_night, is_rush, city_prior "
        "FROM places WHERE place_id=?",
        (place_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"No place row for place_id={place_id}")
    road_kind, is_rain, is_night, is_rush, city_prior = row
    return {
        "road_kind": road_kind,
        "is_rain": bool(is_rain),
        "is_night": bool(is_night),
        "is_rush": bool(is_rush),
        "city_prior": city_prior if city_prior is not None else 1.0,
    }


# ---------------------------------------------------------------------------
# Term computation
# ---------------------------------------------------------------------------

def compute_V(w: Dict[str, float], n_video: Dict[str, int], T: float) -> float:
    return sum(w[k] * n_video[k] / T for k in VIDEO_TYPES)


def compute_P(n_inferred: Dict[str, int], T: float) -> float:
    return sum(n_inferred[k] for k in INFERRED_TYPES) / T


def compute_E(vehicle_count: int, T: float) -> float:
    return vehicle_count / T


def compute_C(place: Dict[str, object], rain_override: Optional[bool]) -> float:
    is_rain = place["is_rain"] if rain_override is None else rain_override
    c = 1.0
    if is_rain:
        c *= C_RAIN
    if place["is_night"]:
        c *= C_NIGHT
    if place["is_rush"]:
        c *= C_RUSH
    if place["road_kind"] == "highway":
        c *= C_HIGHWAY
    c *= float(place["city_prior"])
    return c


def top_types_for(
    w: Dict[str, float], n_video: Dict[str, int], n_inferred: Dict[str, int], T: float
) -> List[str]:
    contributions: Dict[str, float] = {}
    for k in VIDEO_TYPES:
        contributions[k] = A_V * w[k] * n_video[k] / T
    for k in INFERRED_TYPES:
        contributions[k] = B_P * n_inferred[k] / T
    ranked = sorted(contributions.items(), key=lambda kv: -kv[1])
    return [k for k, _ in ranked[:2]]


# ---------------------------------------------------------------------------
# Output table
# ---------------------------------------------------------------------------

def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS net_risk_scores (
            score_id TEXT PRIMARY KEY,
            place_id TEXT NOT NULL,
            window TEXT NOT NULL,
            V REAL, P REAL, E REAL, C REAL,
            R REAL, band TEXT, top_types TEXT,
            vehicle_count INTEGER,
            UNIQUE(place_id, window)
        )
        """
    )


def upsert_score(
    conn: sqlite3.Connection,
    place_id: str,
    window: str,
    V: float, P: float, E: float, C: float,
    R: float, band: str, top_types: List[str], vehicle_count: int,
) -> None:
    conn.execute(
        "DELETE FROM net_risk_scores WHERE place_id=? AND window=?", (place_id, window)
    )
    conn.execute(
        "INSERT INTO net_risk_scores "
        "(score_id, place_id, window, V, P, E, C, R, band, top_types, vehicle_count) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), place_id, window, V, P, E, C, R, band,
         ",".join(top_types), vehicle_count),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def score_window(
    conn: sqlite3.Connection,
    place_id: str,
    source_id: str,
    window: str,
    rain_override: Optional[bool] = None,
) -> Dict[str, object]:
    place = load_place(conn, place_id)
    T = compute_T(conn, place_id, source_id)
    w = compute_weights(S_K)

    n_video = count_types(conn, place_id, source_id, "video", VIDEO_TYPES)
    n_inferred = count_types(conn, place_id, source_id, "inferred", INFERRED_TYPES)
    vehicle_count = count_exposure(conn, place_id, source_id)

    V = compute_V(w, n_video, T)
    P = compute_P(n_inferred, T)
    E = compute_E(vehicle_count, T)
    C = compute_C(place, rain_override)

    R = max(0.0, min(100.0, A_V * V + B_P * P + C_E * E + D_C * C))
    band = band_for(R)
    tops = top_types_for(w, n_video, n_inferred, T)

    ensure_table(conn)
    upsert_score(conn, place_id, window, V, P, E, C, R, band, tops, vehicle_count)

    return {"w": w, "T": T, "V": V, "P": P, "E": E, "C": C,
            "R": R, "band": band, "top_types": tops, "vehicle_count": vehicle_count}


def print_result(label: str, r: Dict[str, object]) -> None:
    print(f"=== {label} ===")
    print(f"weights: { {k: round(v, 3) for k, v in r['w'].items()} }")
    print(f"T={r['T']:.3f} min  V={r['V']:.3f}  P={r['P']:.3f}  "
          f"E={r['E']:.3f}  C={r['C']:.3f}")
    print(f"R={r['R']:.2f}  band={r['band']}  top_types={r['top_types']}")
    print(f"vehicle_count={r['vehicle_count']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: compute net risk score")
    parser.add_argument("--db", required=True, help="Path to sqlite schema.db")
    parser.add_argument("--place", required=True, help="place_id")
    parser.add_argument("--source", required=True, help="source_id (this clip)")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        off = score_window(conn, args.place, args.source, "this_clip", rain_override=False)
        print_result("rain OFF (this_clip)", off)

        on = score_window(conn, args.place, args.source, "this_clip_rain", rain_override=True)
        print_result("rain ON (this_clip_rain)", on)

        print(f"R moved from {off['R']:.2f} ({off['band']}) "
              f"to {on['R']:.2f} ({on['band']})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()