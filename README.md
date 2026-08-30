# Context-Aware Traffic Risk Intelligence & Planning System

One camera feed → detected events → a per-place risk score → two dispatch cards
(police / ambulance). The shared store is a Supabase (Postgres) project so the
whole team can read and write to one database. Stage work lives elsewhere;
this is not that spec.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
cp .env.example .env        # then fill in SUPABASE_URL and SUPABASE_KEY
```

Get `SUPABASE_URL` and `SUPABASE_KEY` from the project's dashboard under
Project Settings -> API. Never commit `.env`.

```bash
python supabase_client.py   # sanity check: confirms the client can connect
```

Tables (`places`, `cleaned_inputs`, `tracked_objects`, `incidents`,
`net_risk_scores`, `scenarios`) are created and managed directly in the
Supabase project, not from this repo.

## Who writes which tables

- **Stage 1** (ingest/cleaning): `cleaned_inputs`, `places`
- **Stage 2** (detection/tracking): `tracked_objects`
- **Stage 3** (incidents/scoring): `incidents`, `net_risk_scores`
- **Stage 4** (dispatch/scenarios): `scenarios`

## Frozen type names

- `incidents.type`: `wrong_way | lane_cut | red_light | speeding | near_miss | harsh_brake`
- `incidents.source`: `video | inferred`

`places.has_signal = 0` means a place cannot produce a `red_light` incident.

## Where the real specs are

Teammates: use `STAGE_1_STARTER.md`, `STAGE_2_STARTER.md`, etc. for your stage.
This README is not a spec.

DB password - Rag398S0TGGpmhFC