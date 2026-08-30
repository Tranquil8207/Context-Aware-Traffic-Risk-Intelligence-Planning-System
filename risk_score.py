"""Stage 3: net risk score for one (place, clip) pair.

Read-only against places / incidents / tracked_objects; writes exactly one
row into net_risk_scores, upserted on (place_id, window) -- run the
migration in add_net_risk_scores_unique.sql once before first use.

S_K (MoRTH persons-killed shares per incident type) and the a/b/c/d blend
weights are deliberately NOT hardcoded here -- they're collected by the
UI's Risk tab (after Events) and passed in by the caller.

Split into two phases so a/b/c/d can be recomputed live in the UI without
a DB round-trip on every change:
  - compute_metrics() -- Steps 1-7. Needs the DB (place lookup, incident
    counts, distinct vehicles) and s_k (V depends on w_k <- s_k). Call this
    once per (source, s_k) combination.
  - combine_risk() -- Step 8. Pure arithmetic on compute_metrics()'s
    output plus a/b/c/d. No DB access -- safe to call on every slider tick.
  - save_risk_row() -- Step 9. The actual upsert, once the user wants to
    commit the current values.
  - compute_risk() -- convenience wrapper chaining all three, for simple
    one-shot callers that don't need the live-recompute split.
"""

import math
from typing import Any

VIDEO_TYPES = ("speeding", "wrong_way", "red_light", "lane_cut")
INFERRED_TYPES = ("near_miss", "harsh_brake", "weave")


def compute_weights(s_k: dict[str, float]) -> dict[str, float]:
    """w_k = ln(1 + s_k) / sum(ln(1 + s_j)) -- normalized over whichever
    keys are actually in s_k, so leaving unused types (e.g. red_light,
    lane_cut) in or out is the caller's choice, not this function's."""

    logs = {k: math.log(1 + v) for k, v in s_k.items()}
    total = sum(logs.values())

    if total <= 0:
        return {k: 0.0 for k in s_k}

    return {k: v / total for k, v in logs.items()}


def _incident_counts(
    client, place_id: int, source_id: str, source: str, use_conf: bool
) -> dict[str, float]:
    response = (
        client.table("incidents")
        .select("type, conf")
        .eq("place_id", place_id)
        .eq("source_id", source_id)
        .eq("source", source)
        .execute()
    )

    counts: dict[str, float] = {}
    for row in response.data:
        key = row["type"]
        weight = float(row["conf"]) if use_conf else 1.0
        counts[key] = counts.get(key, 0.0) + weight

    return counts


def compute_metrics(
    client,
    place_id: int,
    source_id: str,
    duration_s: float,
    s_k: dict[str, float],
    use_conf: bool = False,
) -> dict[str, Any]:
    """Steps 1-7: everything that touches the DB, independent of a/b/c/d.

    use_conf=True swaps COUNT(*)-style counting for SUM(conf) (innovation
    item A) -- off by default since incidents.conf is currently a flat 1.0
    for every row in this codebase, making the swap a no-op either way
    until real per-incident confidence values exist.
    """

    # Step 1 -- place context
    place_response = (
        client.table("places").select("*").eq("place_id", place_id).execute()
    )
    if not place_response.data:
        raise ValueError(f"No place row for place_id={place_id}")
    place = place_response.data[0]

    # Step 2 -- clip length T, in minutes, floored so division never breaks
    T = max(duration_s / 60.0, 1.0 / 60.0)

    # Step 3 -- weights, printed per the spec ("fixed per run")
    w_k = compute_weights(s_k)
    print(f"[risk_score] w_k = {w_k}")

    # Step 4 -- V (video-sourced: speeding, wrong_way; red_light/lane_cut
    # are not emitted by this project's Stage 2, so their n_k is always 0)
    video_counts = _incident_counts(client, place_id, source_id, "video", use_conf)
    V = sum(
        w_k.get(k, 0.0) * (video_counts.get(k, 0.0) / T) for k in VIDEO_TYPES
    )

    # Step 5 -- P (inferred: near_miss, harsh_brake, weave -- weave folded
    # in as a core term here, not held back as future work, since it's
    # already one of this project's five always-on Stage 2 event types)
    inferred_counts = _incident_counts(client, place_id, source_id, "inferred", use_conf)
    n_inferred = sum(inferred_counts.get(k, 0.0) for k in INFERRED_TYPES)
    P = n_inferred / T

    # Step 6 -- E (distinct tracked vehicles / minute)
    tracks_response = (
        client.table("tracked_objects")
        .select("track_id")
        .eq("place_id", place_id)
        .eq("source_id", source_id)
        .execute()
    )
    vehicle_count = len({row["track_id"] for row in tracks_response.data})
    E = vehicle_count / T

    # Step 7 -- C (place multipliers)
    C = 1.0
    if place.get("is_rain"):
        C *= 1.15
    if place.get("is_night"):
        C *= 1.10
    if place.get("is_rush"):
        C *= 1.10
    if place.get("road_kind") == "highway":
        C *= 1.10
    C *= place.get("city_prior") or 1.0

    return {
        "V": V,
        "P": P,
        "E": E,
        "C": C,
        "T": T,
        "w_k": w_k,
        "video_counts": video_counts,
        "inferred_counts": inferred_counts,
        "vehicle_count": vehicle_count,
    }


def combine_risk(metrics: dict[str, Any], a: float, b: float, c: float, d: float) -> dict[str, Any]:
    """Step 8: pure arithmetic on compute_metrics()'s output. No DB access
    -- this is what the UI calls on every a/b/c/d change for live R updates."""

    V, P, E, C = metrics["V"], metrics["P"], metrics["E"], metrics["C"]
    T = metrics["T"]
    w_k = metrics["w_k"]
    video_counts = metrics["video_counts"]
    inferred_counts = metrics["inferred_counts"]

    R = max(0.0, min(100.0, a * V + b * P + c * E + d * C))
    band = "cold" if R < 40 else "warm" if R < 70 else "hot"

    contributions = {
        k: a * w_k.get(k, 0.0) * video_counts.get(k, 0.0) / T for k in VIDEO_TYPES
    }
    contributions.update(
        {k: b * inferred_counts.get(k, 0.0) / T for k in INFERRED_TYPES}
    )
    top_types = ",".join(
        k
        for k, _ in sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)[:2]
    )

    return {
        "V": V,
        "P": P,
        "E": E,
        "C": C,
        "R": R,
        "band": band,
        "top_types": top_types,
        "vehicle_count": metrics["vehicle_count"],
    }


def save_risk_row(client, place_id: int, window: str, row: dict[str, Any]) -> dict[str, Any]:
    """Step 9: write, keyed on (place_id, window)."""

    full_row = {"place_id": place_id, "window": window, **row}
    (
        client.table("net_risk_scores")
        .upsert(full_row, on_conflict="place_id,window")
        .execute()
    )
    return full_row


def compute_risk(
    client,
    place_id: int,
    source_id: str,
    duration_s: float,
    s_k: dict[str, float],
    a: float,
    b: float,
    c: float,
    d: float,
    window: str = "this_clip",
    use_conf: bool = False,
) -> dict[str, Any]:
    """Convenience one-shot: metrics + combine + save. For callers that
    don't need the live-recompute split (e.g. an explicit Save action, or
    a script running the whole pipeline non-interactively)."""

    metrics = compute_metrics(client, place_id, source_id, duration_s, s_k, use_conf)
    row = combine_risk(metrics, a, b, c, d)
    return save_risk_row(client, place_id, window, row)
