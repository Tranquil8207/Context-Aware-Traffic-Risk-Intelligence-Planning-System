-- Context-Aware Traffic Risk Intelligence & Planning System
-- Reference schema for the Supabase (Postgres) project.
-- This is not run by any script in this repo — paste it into the Supabase
-- SQL Editor to create the tables. Do not add tables/columns not listed here
-- without updating this file to match.

CREATE TABLE IF NOT EXISTS places (
    place_id         bigint generated always as identity primary key,
    lat              double precision,
    lng              double precision,
    road_kind        text,
    speed_limit_kmh  double precision,
    ref_length_m     double precision,
    ref_points_json  jsonb,
    stop_line_json   jsonb,
    legal_heading    double precision,
    lanes_json       jsonb,
    has_signal       smallint NOT NULL DEFAULT 0,  -- 0/1; 0 means no red_light incidents here
    city_prior       double precision,
    is_night         smallint,
    is_rain          smallint,
    is_rush          smallint
);

CREATE TABLE IF NOT EXISTS cleaned_inputs (
    source_id   text PRIMARY KEY,
    kind        text,
    place_id    bigint REFERENCES places(place_id),
    file_name   text,
    clock_start text,
    clock_note  text,
    notes       text
);

CREATE TABLE IF NOT EXISTS tracked_objects (
    track_id       bigint NOT NULL,
    source_id      text NOT NULL REFERENCES cleaned_inputs(source_id),
    place_id       bigint REFERENCES places(place_id),
    class          text,
    path_json      jsonb,
    mean_speed_kmh double precision,
    object_risk    double precision,
    PRIMARY KEY (source_id, track_id)
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id bigint generated always as identity primary key,
    source_id   text,
    place_id    bigint REFERENCES places(place_id),
    track_id    bigint,
    ts_ms       bigint,
    type        text NOT NULL CHECK (type IN (
                    'wrong_way', 'lane_cut', 'red_light',
                    'speeding', 'near_miss', 'harsh_brake'
                )),
    source      text NOT NULL CHECK (source IN ('video', 'inferred')),
    conf        double precision,
    meta_json   jsonb,
    FOREIGN KEY (source_id, track_id) REFERENCES tracked_objects(source_id, track_id)
);

-- "window" is a reserved word in Postgres (window functions) — quote it as
-- "window" when referencing the column in queries.
CREATE TABLE IF NOT EXISTS net_risk_scores (
    score_id      bigint generated always as identity primary key,
    place_id      bigint REFERENCES places(place_id),
    "window"      text,
    "V"           double precision,
    "P"           double precision,
    "E"           double precision,
    "C"           double precision,
    "R"           double precision,
    band          text,
    top_types     text,
    vehicle_count integer
);

CREATE TABLE IF NOT EXISTS scenarios (
    run_id         bigint generated always as identity primary key,
    score_id       bigint REFERENCES net_risk_scores(score_id),
    intervention   text,
    "R_before"     double precision,
    "R_after"      double precision,
    note_police    text,
    note_ambulance text
);
