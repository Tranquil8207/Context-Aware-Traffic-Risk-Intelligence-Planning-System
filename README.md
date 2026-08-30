# Context-Aware Traffic Risk Intelligence & Planning System

One traffic camera clip → calibrated real-world coordinates → detected driving
events → a per-clip risk score. Everything runs through a small web UI
(**catris**) that walks a video through five steps: **Ingest → Homography →
Lanes → Events → Risk**. The shared store is a Supabase (Postgres) project so
the whole team reads and writes the same database.

This README describes the system as it actually works today, not as
originally spec'd — a lot has changed since the first draft (SQLite → Supabase,
automatic YOLOP lane segmentation → manual lane boxing, a CLI-only pipeline →
a full web UI, three detected event types → five).

## What's in this repo

| Piece | Files |
|---|---|
| Web UI backend | `app.py`, `source_io.py` |
| Web UI frontend | `static/index.html`, `static/app.js`, `static/app.css` |
| Detection & tracking | `objectdetection.py` (YOLO + ByteTrack, homography, lane lookup, event detection) |
| Calibration tools (standalone or UI-driven) | `calibrate_homography.py`, `calibrate_lanes.py`, `calibration_io.py` |
| Risk scoring (Stage 3) | `risk_score.py` |
| Database | `schema.sql` (reference DDL — paste into Supabase), `supabase_client.py` |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env        # then fill in SUPABASE_URL and SUPABASE_KEY
```

Get `SUPABASE_URL` and `SUPABASE_KEY` from the Supabase dashboard under
Project Settings -> API. Never commit `.env`.

Then create the tables: open the Supabase SQL Editor and run `schema.sql`
once. It's a reference file — nothing in this repo runs it for you.

```bash
python supabase_client.py   # sanity check: confirms the client can connect
```

Get the object detector's weights (`yolo26m.pt`, referenced by
`objectdetection.py`'s `OBJECT_MODEL_PATH`) and place the file at the repo
root. There is no automatic lane-segmentation model anymore — lanes are
boxed by hand in the UI (see below), so nothing else needs downloading.

## Running it

```bash
python app.py
# then open http://localhost:8000
```

Everything — ingesting a video, calibrating it, tuning detection thresholds,
running detection, and scoring risk — happens through this one page.

## How to use it: the five steps

**1. Ingest**
Drop in a video file and fill in what you already know about the place: the
posted speed limit, road type, and whether the clip is at night / in rain /
during rush hour. This writes a `places` + `cleaned_inputs` row and captures
the video's actual duration automatically.

**2. Homography**
Click 4 points on a frame — near-left, near-right, far-left, far-right —
tracing two reference lines across the road at two different distances from
the camera. Then enter the *real* measurements those points represent: the
lane width (or however many lanes you spanned) and the distance between the
near and far lines. This turns pixel coordinates into real-world meters for
everything downstream — position, speed, distance-based checks. Get the real
numbers from something you can actually verify (a car's known length, road
marking dash/gap standards, a satellite measurement) — never guess. This
whole pipeline is only as good as these two numbers.

**3. Lanes**
Trace each visible lane as a polygon — click as many points as needed per
lane, so curves and merges are no extra work. After finishing a lane's
boundary, optionally click a 2-point arrow showing that lane's legal
direction of travel; it gets converted through the homography into a real
heading, which is what wrong-way-driving detection compares against. Skip
the arrow (press Enter) if you don't need heading for that lane.

**4. Events**
Tune the thresholds for the five things this system detects (see below), or
just accept the defaults. Some — like the harsh-braking deceleration
threshold — are sourced from published research; others (`SPEED_LIMIT_KMH`,
`V_MIN_KMH`) are specific to the road in your clip and need a real value
before their event type can ever fire.

**5. Risk**
Click "Identify" to actually run detection on the video (YOLO + ByteTrack
tracking, using everything calibrated in steps 2-4). Once it finishes, tune
the risk formula's inputs — MoRTH-sourced severity shares per incident type,
and four blend weights — and watch the risk score `R` and its cold/warm/hot
band update live in the browser as you adjust them. Nothing is written to
the database until you press Save.

## What gets detected

Five event types, matching a locked list — no others are emitted:

| Type | Source | What it means |
|---|---|---|
| `wrong_way` | video | Heading deviates from the lane's legal direction beyond a threshold, sustained for a dwell period |
| `speeding` | video | Mean speed over a short window is above `SPEED_LIMIT_KMH` (`meta.kind="over"`) or below `V_MIN_KMH` while in a mapped lane (`meta.kind="under"`) |
| `harsh_brake` | inferred | Longitudinal speed drops sharply within a short window, only counted once a track has enough history and was moving fast enough beforehand |
| `near_miss` | inferred | Two tracks' real-world positions stay closer than a distance threshold for several consecutive frames |
| `weave` | inferred | Sustained or oscillating lateral movement beyond a threshold |

`red_light` and `lane_cut` exist as valid database values (see below) but
aren't emitted by anything in this pass.

## Database schema

Six tables, all in `schema.sql`:

- **`places`** — one row per physical location/camera setup: coordinates, road type, calibration reference data, lane polygons, and condition flags (rain/night/rush).
- **`cleaned_inputs`** — one row per ingested video, linked to a place.
- **`tracked_objects`** — one row per tracked vehicle per video, with its full path (`[t_ms, x, y, lane]` per point) and mean speed.
- **`incidents`** — one row per detected event. `type` and `source` are frozen enums (see below) enforced by a `CHECK` constraint.
- **`net_risk_scores`** — one row per (place, window), upserted — re-scoring the same clip overwrites rather than duplicates.
- **`scenarios`** — reserved for later "what-if" comparisons; nothing in this pipeline writes to it yet.

Frozen values, enforced at the database level:
- `incidents.type`: `wrong_way | lane_cut | red_light | speeding | near_miss | harsh_brake | weave`
- `incidents.source`: `video | inferred`

`places.has_signal = 0` means a place cannot produce a `red_light` incident
(not currently emitted regardless, per above).

## The risk score

`R = clamp(a·V + b·P + c·E + d·C, 0, 100)`, banded `cold` (<40) / `warm`
(<70) / `hot`. `V` and `P` are incident rates (video-sourced and
inferred-sourced) weighted by MoRTH real-world accident-severity shares;
`E` is vehicle throughput; `C` is a place-condition multiplier (rain, night,
rush, road type). The `a/b/c/d` blend weights and the severity shares are
UI inputs with sensible defaults (see `source_io.py`'s `RISK_S_K_DEFAULTS`
and `RISK_ABCD_DEFAULTS`), not hardcoded — recompute is instant in the
browser as you adjust them, since only fetching the base numbers needs the
database.

## Notes for anyone extending this

- Calibration data (homography + lanes) lives in each source's own record under `data/sources/<id>/ingest.json`, not committed to git — recalibrating a new video never means editing code.
- `objectdetection.py` reads its configuration from that same ingest record via `apply_source()` — nothing needed for a specific clip is hardcoded in the file.
- The standalone calibration tools (`calibrate_homography.py`, `calibrate_lanes.py`) still work outside the UI if you'd rather run them from the command line; they write to the same `calibration.json` format.
