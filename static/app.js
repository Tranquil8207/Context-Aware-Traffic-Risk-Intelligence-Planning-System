(() => {
  const $ = (id) => document.getElementById(id);

  const HOMO_LABELS = ["near-left", "near-right", "far-left", "far-right"];
  const HOMO_COLORS = ["#22c55e", "#22c55e", "#f59e0b", "#f59e0b"];
  const LANE_COLORS = ["#3b82f6", "#f59e0b", "#e879f9", "#22c55e", "#22d3ee", "#ef4444"];
  const EVENT_KEYS = [
    "CONFIDENCE",
    "SPEED_SMOOTHING_FRAMES",
    "MAX_REASONABLE_SPEED_KMH",
    "WRONG_WAY_ANGLE_DEG",
    "WRONG_WAY_DWELL_S",
    "SPEEDING_OVER_WINDOW_S",
    "SPEEDING_UNDER_WINDOW_S",
    "HARSH_BRAKE_DROP_KMH",
    "HARSH_BRAKE_WINDOW_S",
    "HARSH_BRAKE_MIN_SPEED_KMH",
    "NEAR_MISS_GAP_M",
    "NEAR_MISS_MIN_FRAMES",
    "WEAVE_VLAT_LIM_KMH",
    "WEAVE_WINDOW_FRAMES",
    "V_MIN_KMH",
  ];
  const RISK_SK_KEYS = ["speeding", "wrong_way", "red_light", "lane_cut"];
  const RISK_WEIGHT_KEYS = ["a", "b", "c", "d"];
  const EVENT_DEFAULTS = {
    CONFIDENCE: 0.35,
    SPEED_SMOOTHING_FRAMES: 10,
    MAX_REASONABLE_SPEED_KMH: 180,
    WRONG_WAY_ANGLE_DEG: 150,
    WRONG_WAY_DWELL_S: 4,
    SPEEDING_OVER_WINDOW_S: 0.5,
    SPEEDING_UNDER_WINDOW_S: 1,
    HARSH_BRAKE_DROP_KMH: 7,
    HARSH_BRAKE_WINDOW_S: 1,
    HARSH_BRAKE_MIN_SPEED_KMH: 20,
    NEAR_MISS_GAP_M: 2,
    NEAR_MISS_MIN_FRAMES: 3,
    WEAVE_VLAT_LIM_KMH: 6,
    WEAVE_WINDOW_FRAMES: 5,
  };

  const state = {
    file: null,
    objectUrl: null,
    selectedId: null,
    sources: [],
    record: null,
    stage: "ingest",
    frameIndex: 0,
    frameCount: 0,
    frameWidth: 0,
    frameHeight: 0,
    frameImage: null,
    homoPoints: [],
    finishedLanes: [],
    currentPolygon: [],
    pendingPolygon: null,
    currentArrow: [],
    riskMetrics: null,
    riskMetricsReq: 0,
    riskSkTimer: null,
    popup: null,
    popupCanvas: null,
  };

  const canvas = $("calibCanvas");
  let ctx = canvas.getContext("2d");

  function log(message) {
    const ts = new Date().toISOString().replace("T", " ").slice(0, 19);
    $("log").textContent += `[${ts}] ${message}\n`;
    $("log").scrollTop = $("log").scrollHeight;
  }

  function setStatus(text) {
    $("statusText").textContent = text;
  }

  function setTitle(name) {
    $("windowTitle").textContent = `catris v0.1  —  ${name || "[untitled]"}`;
  }

  function stemFromName(name) {
    return (name || "source").replace(/\.[^.]+$/, "").replace(/\s+/g, "_");
  }

  function closeMenus() {
    document.querySelectorAll(".menu.open").forEach((el) => el.classList.remove("open"));
  }

  function typing() {
    const tag = (document.activeElement && document.activeElement.tagName) || "";
    return tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";
  }

  function eventsComplete(record) {
    if (!record) return false;
    if (record.stage === "ready" || record.stage === "risk") return true;
    return Boolean(record.run && record.run.status === "ok");
  }

  function nextStage(record) {
    if (!record) return "ingest";
    if (!record.homography) return "homography";
    if (!record.lanes || !record.lanes.length) return "lanes";
    if (!eventsComplete(record)) return "events";
    if (record.risk) return "risk";
    return "events";
  }

  function setStage(stage) {
    state.stage = stage;
    $("panelIngest").hidden = stage !== "ingest";
    $("panelHomography").hidden = stage !== "homography";
    $("panelLanes").hidden = stage !== "lanes";
    $("panelEvents").hidden = stage !== "events";
    $("panelRisk").hidden = stage !== "risk";
    $("tabIngest").classList.toggle("on", stage === "ingest");
    $("tabHomography").classList.toggle("on", stage === "homography");
    $("tabLanes").classList.toggle("on", stage === "lanes");
    $("tabEvents").classList.toggle("on", stage === "events");
    $("tabRisk").classList.toggle("on", stage === "risk");
    document.querySelectorAll("#menu-module [data-goto]").forEach((btn) => {
      btn.classList.toggle("checked", btn.dataset.goto === stage);
    });
    const labels = {
      ingest: "Step 1 / 5  —  Ingest",
      homography: "Step 2 / 5  —  Homography",
      lanes: "Step 3 / 5  —  Lanes",
      events: "Step 4 / 5  —  Events",
      risk: "Step 5 / 5  —  Risk",
    };
    $("moduleChip").textContent = labels[stage] || stage;
    const showCanvas = stage === "homography" || stage === "lanes" || stage === "events" || stage === "risk";
    $("preview").hidden = showCanvas || !(state.file || state.selectedId);
    canvas.hidden = !showCanvas;
    $("emptyView").hidden = showCanvas || Boolean(state.file) || Boolean(state.selectedId);
    $("prevFrameBtn").disabled = !showCanvas;
    $("nextFrameBtn").disabled = !showCanvas;
    if (!showCanvas) $("viewHud").textContent = "";
    if (showCanvas && state.selectedId) loadFrame(state.frameIndex);
    if (stage === "homography" || stage === "lanes") {
      openCalibPopup();
      renderHomoTable();
      if (stage === "lanes") renderLaneTable();
    } else {
      closeCalibPopup();
    }
    if (stage === "events") {
      fillEventForm(state.record);
      loadIncidents();
    }
    if (stage === "risk") loadRisk();
  }

  function unlockLaterSteps(record) {
    const hasSource = Boolean(record);
    const hasHomo = Boolean(record && record.homography);
    const hasLanes = Boolean(record && record.lanes && record.lanes.length);
    $("tabHomography").disabled = !hasSource;
    $("tabLanes").disabled = !hasHomo;
    $("tabEvents").disabled = !hasLanes;
    $("tabRisk").disabled = !eventsComplete(record);
    $("menuHomography").disabled = !hasSource;
    $("menuLanes").disabled = !hasHomo;
    $("menuEvents").disabled = !hasLanes;
    $("menuRisk").disabled = !eventsComplete(record);
  }

  function setPreviewFile(file) {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.file = file || null;
    state.objectUrl = file ? URL.createObjectURL(file) : null;
    $("filePath").textContent = file ? file.name : "(no video)";
    if (file) {
      $("preview").src = state.objectUrl;
      $("preview").hidden = state.stage !== "ingest";
      $("emptyView").hidden = true;
      setTitle(file.name);
      $("statusSource").textContent = file.name;
      log(`Loaded local video: ${file.name}`);
    } else {
      $("preview").removeAttribute("src");
      $("preview").hidden = true;
      $("emptyView").hidden = state.stage !== "ingest";
      setTitle("[untitled]");
      $("statusSource").textContent = "no source";
    }
  }

  function resetProject() {
    $("ingestForm").reset();
    $("speed_limit_kmh").value = "80";
    $("city_prior").value = "1.0";
    $("road_kind").value = "expressway";
    state.selectedId = null;
    state.record = null;
    state.frameIndex = 0;
    state.frameCount = 0;
    state.frameImage = null;
    state.homoPoints = [];
    state.finishedLanes = [];
    state.currentPolygon = [];
    state.pendingPolygon = null;
    state.currentArrow = [];
    setPreviewFile(null);
    closeCalibPopup();
    resetRiskForm();
    unlockLaterSteps(null);
    setStage("ingest");
    setStatus("Ready");
    document.querySelectorAll(".tree-item").forEach((el) => el.classList.remove("active"));
    log("New project.");
  }

  function renderTree(sources) {
    const host = $("sourceTree");
    host.innerHTML = "";
    if (!sources.length) {
      host.className = "tree-empty";
      host.textContent = "No sources ingested";
      return;
    }
    host.className = "";
    sources.forEach((src) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "tree-item" + (src.source_id === state.selectedId ? " active" : "");
      btn.textContent = `${src.source_id}  (${src.stage || "ingest"})`;
      btn.addEventListener("click", () => openSource(src.source_id));
      host.appendChild(btn);
    });
  }

  async function loadSources() {
    const res = await fetch("/api/sources");
    const data = await res.json();
    state.sources = data.sources || [];
    renderTree(state.sources);
  }

  function applyRecord(record) {
    state.record = record;
    state.selectedId = record.source_id;
    $("speed_limit_kmh").value = record.place?.speed_limit_kmh ?? 80;
    $("v_min_kmh").value = record.processing?.v_min_kmh ?? "";
    $("road_kind").value = record.place?.road_kind || "expressway";
    $("lat").value = record.place?.lat ?? "";
    $("lng").value = record.place?.lng ?? "";
    $("city_prior").value = record.place?.city_prior ?? 1;
    $("is_night").checked = Boolean(record.place?.is_night);
    $("is_rain").checked = Boolean(record.place?.is_rain);
    $("is_rush").checked = Boolean(record.place?.is_rush);
    $("has_signal").checked = Boolean(record.place?.has_signal);
    $("filePath").textContent = record.file_name || record.source_id;
    if (record.homography) {
      state.homoPoints = (record.homography.src || []).map((pt) => ({ x: pt[0], y: pt[1] }));
      $("lane_width_m").value = record.homography.lane_width_m ?? 3.5;
      $("reference_distance_m").value = record.homography.reference_distance_m ?? 20;
      state.frameIndex = record.homography.frame_index || 0;
    } else {
      state.homoPoints = [];
    }
    if (record.lanes && record.lanes.length) {
      const arrows = record.lane_arrows || [];
      state.finishedLanes = record.lanes.map((lane, i) => ({
        polygon: lane.polygon.map(([x, y]) => ({ x, y })),
        arrow: arrows[i] ? { start: { x: arrows[i][0][0], y: arrows[i][0][1] }, end: { x: arrows[i][1][0], y: arrows[i][1][1] } } : null,
        heading: lane.heading,
      }));
    } else {
      state.finishedLanes = [];
    }
    state.currentPolygon = [];
    state.pendingPolygon = null;
    state.currentArrow = [];
    const db = record.supabase && record.supabase.ok ? `place ${record.place_id}` : "local only";
    $("statusDb").textContent = `supabase: ${db}`;
    $("statusSource").textContent = record.source_id;
    setTitle(record.file_name || record.source_id);
    unlockLaterSteps(record);
  }

  async function openSource(sourceId) {
    const res = await fetch(`/api/sources/${encodeURIComponent(sourceId)}`);
    if (!res.ok) {
      log(`Could not open ${sourceId}`);
      return;
    }
    const record = await res.json();
    state.file = null;
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
    applyRecord(record);
    await loadSources();
    setStage(nextStage(record));
    log(`Opened source ${sourceId} (stage: ${record.stage})`);
  }

  function collectIngest() {
    const data = new FormData();
    data.append("video", state.file, state.file.name);
    data.append("source_id", stemFromName(state.file.name));
    data.append("lat", $("lat").value);
    data.append("lng", $("lng").value);
    data.append("road_kind", $("road_kind").value);
    data.append("speed_limit_kmh", $("speed_limit_kmh").value);
    data.append("v_min_kmh", $("v_min_kmh").value);
    data.append("has_signal", $("has_signal").checked ? "1" : "0");
    data.append("is_night", $("is_night").checked ? "1" : "0");
    data.append("is_rain", $("is_rain").checked ? "1" : "0");
    data.append("is_rush", $("is_rush").checked ? "1" : "0");
    data.append("city_prior", $("city_prior").value);
    return data;
  }

  function ingestSource() {
    if (!state.file) {
      setStatus("No video selected");
      log("Ingest aborted: choose a video first.");
      return;
    }
    const xhr = new XMLHttpRequest();
    $("progress").hidden = false;
    $("progressBar").style.width = "0%";
    $("ingestBtn").disabled = true;
    setStatus("Uploading…");
    xhr.upload.onprogress = (ev) => {
      if (!ev.lengthComputable) return;
      const pct = Math.round((ev.loaded / ev.total) * 100);
      $("progressBar").style.width = `${pct}%`;
      setStatus(`Uploading… ${pct}%`);
    };
    xhr.onload = async () => {
      $("ingestBtn").disabled = false;
      $("progress").hidden = true;
      let payload = null;
      try { payload = JSON.parse(xhr.responseText); } catch { payload = { detail: xhr.responseText }; }
      if (xhr.status >= 400) {
        setStatus("Ingest failed");
        log(`Ingest failed: ${payload.detail || xhr.statusText}`);
        return;
      }
      applyRecord(payload);
      if (payload.supabase && payload.supabase.ok) {
        log(`Wrote places.place_id=${payload.place_id} and cleaned_inputs.${payload.source_id}`);
      } else {
        log(`Saved locally. Supabase skipped: ${(payload.supabase && payload.supabase.error) || "not connected"}`);
      }
      setStatus("Ingest complete — calibrate homography");
      log(`Ingest complete: ${payload.source_id}. Click near-left, near-right, far-left, far-right.`);
      await loadSources();
      setStage("homography");
    };
    xhr.onerror = () => {
      $("ingestBtn").disabled = false;
      $("progress").hidden = true;
      setStatus("Network error");
      log("Ingest failed: network error");
    };
    xhr.open("POST", "/api/ingest");
    xhr.send(collectIngest());
  }

  function calibCanvases() {
    const list = [canvas];
    if (state.popupCanvas && state.popup && !state.popup.closed) list.push(state.popupCanvas);
    return list;
  }

  function setHud(text) {
    $("viewHud").textContent = text;
    if (state.popup && !state.popup.closed) {
      const hud = state.popup.document.getElementById("hud");
      if (hud) hud.textContent = text;
    }
  }

  function syncPopupChrome() {
    const open = Boolean(state.popup && !state.popup.closed);
    $("openFrameBtn").hidden = !((state.stage === "homography" || state.stage === "lanes") && !open);
    if (!open) return;
    const title = state.popup.document.getElementById("title");
    const hint = state.popup.document.getElementById("hint");
    const frame = state.popup.document.getElementById("frame");
    if (title) {
      title.textContent = state.stage === "lanes"
        ? "Lane boxing"
        : "Homography — near-left, near-right, far-left, far-right";
    }
    if (hint) {
      hint.textContent = state.stage === "lanes"
        ? "click=add point  u=undo  Enter/c=finish/skip heading  z=undo lane  r=reset  n/p=frame"
        : "click=add point  r=reset  n/p=frame  q=close window";
    }
    if (frame) {
      frame.textContent = `frame ${state.frameIndex} / ${Math.max(state.frameCount - 1, 0)}`;
    }
    state.popup.document.title = state.stage === "lanes" ? "catris — lanes" : "catris — homography";
  }

  function closeCalibPopup() {
    if (state.popup && !state.popup.closed) {
      try { state.popup.close(); } catch { /* ignore */ }
    }
    state.popup = null;
    state.popupCanvas = null;
    $("openFrameBtn").hidden = true;
  }

  function openCalibPopup() {
    if (state.stage !== "homography" && state.stage !== "lanes") return;
    if (state.popup && !state.popup.closed) {
      try { state.popup.focus(); } catch { /* ignore */ }
      syncPopupChrome();
      draw();
      return;
    }
    const imgW = state.frameImage ? state.frameImage.width : (state.frameWidth || 1280);
    const imgH = state.frameImage ? state.frameImage.height : (state.frameHeight || 720);
    const w = Math.min(screen.availWidth - 24, Math.max(960, imgW + 24));
    const h = Math.min(screen.availHeight - 48, Math.max(640, imgH + 100));
    const win = window.open(
      "",
      "catrisCalib",
      `popup=yes,width=${w},height=${h},left=40,top=40,menubar=no,toolbar=no,location=no,status=no`,
    );
    if (!win) {
      $("openFrameBtn").hidden = false;
      log("Frame window blocked — click Open frame window (allow popups for this site).");
      return;
    }
    state.popup = win;
    win.document.open();
    win.document.write(`<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<title>catris — calibration</title>
<style>
  html, body { margin: 0; height: 100%; background: #0a0a0a; color: #f8f8ff;
    font: 12px/1.35 "Segoe UI", Tahoma, sans-serif; }
  .wrap { height: 100%; display: grid; grid-template-rows: 32px minmax(0,1fr) 28px; }
  .bar { display: flex; align-items: center; gap: 10px; padding: 0 10px;
    background: linear-gradient(#323232, #242424); border-bottom: 1px solid #111; user-select: none; }
  .bar.bottom { border-bottom: 0; border-top: 1px solid #111; color: #b8b8b8; }
  #title { color: #ffefd5; font-weight: 700; }
  #hud { color: #f0e68c; }
  .stage { min-height: 0; display: grid; place-items: center; overflow: auto; background: #000; }
  canvas { cursor: crosshair; display: block; max-width: 100%; max-height: 100%;
    width: auto; height: auto; object-fit: contain; }
  button { color: #fff8dc; background: linear-gradient(#101010, #202020 50%, #101010);
    border: 1px solid #2e2e2e; border-radius: 6px; padding: 3px 10px; }
  button:hover { border-color: #bdb76b; }
</style>
</head>
<body>
<div class="wrap">
  <div class="bar">
    <span id="title">Calibration</span>
    <span id="hud"></span>
    <span style="margin-left:auto" id="frame">frame 0 / 0</span>
    <button type="button" id="prev">Prev frame</button>
    <button type="button" id="next">Next frame</button>
  </div>
  <div class="stage"><canvas id="c"></canvas></div>
  <div class="bar bottom" id="hint">click=add point</div>
</div>
</body>
</html>`);
    win.document.close();
    const pc = win.document.getElementById("c");
    state.popupCanvas = pc;
    pc.addEventListener("click", onCanvasClick);
    win.document.getElementById("prev").addEventListener("click", () => {
      if (state.frameIndex > 0) loadFrame(state.frameIndex - 1);
    });
    win.document.getElementById("next").addEventListener("click", () => {
      if (state.frameIndex + 1 < state.frameCount) loadFrame(state.frameIndex + 1);
    });
    win.addEventListener("keydown", handleCalibKeys);
    win.addEventListener("unload", () => {
      if (state.popup === win) {
        state.popup = null;
        state.popupCanvas = null;
        if (state.stage === "homography" || state.stage === "lanes") {
          $("openFrameBtn").hidden = false;
        }
      }
    });
    syncPopupChrome();
    draw();
    try { win.focus(); } catch { /* ignore */ }
  }

  async function loadFrame(index) {
    if (!state.selectedId) return;
    const res = await fetch(`/api/sources/${encodeURIComponent(state.selectedId)}/frame?index=${index}`);
    if (!res.ok) {
      log("Could not load frame");
      return;
    }
    state.frameIndex = Number(res.headers.get("X-Frame-Index") || index);
    state.frameCount = Number(res.headers.get("X-Frame-Count") || state.record?.media?.frame_count || 0);
    state.frameWidth = Number(res.headers.get("X-Frame-Width") || 0);
    state.frameHeight = Number(res.headers.get("X-Frame-Height") || 0);
    $("frameLabel").textContent = `${state.frameIndex} / ${Math.max(state.frameCount - 1, 0)}`;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      state.frameImage = img;
      calibCanvases().forEach((c) => {
        c.width = img.width;
        c.height = img.height;
      });
      syncPopupChrome();
      draw();
    };
    img.src = url;
  }

  function canvasPoint(ev) {
    const c = ev.currentTarget || canvas;
    const rect = c.getBoundingClientRect();
    return {
      x: Math.round((ev.clientX - rect.left) * (c.width / rect.width)),
      y: Math.round((ev.clientY - rect.top) * (c.height / rect.height)),
    };
  }

  function draw() {
    if (!state.frameImage) return;
    const saved = ctx;
    calibCanvases().forEach((c) => {
      if (c.width !== state.frameImage.width || c.height !== state.frameImage.height) {
        c.width = state.frameImage.width;
        c.height = state.frameImage.height;
      }
      ctx = c.getContext("2d");
      ctx.drawImage(state.frameImage, 0, 0);
      if (state.stage === "homography") drawHomography();
      if (state.stage === "lanes" || state.stage === "events" || state.stage === "risk") drawLanes();
    });
    ctx = saved;
    if (state.stage === "events") {
      setHud("Lanes frozen — edit event thresholds below, then run identification.");
    }
    if (state.stage === "risk") {
      setHud("Lanes frozen — set s_k and blend weights, then Save.");
    }
    syncPopupChrome();
  }

  function drawHomography() {
    const pts = state.homoPoints;
    pts.forEach((pt, i) => {
      ctx.fillStyle = HOMO_COLORS[i];
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "14px Segoe UI";
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 3;
      ctx.strokeText(`${i + 1}:${HOMO_LABELS[i]}`, pt.x + 8, pt.y - 8);
      ctx.fillStyle = HOMO_COLORS[i];
      ctx.fillText(`${i + 1}:${HOMO_LABELS[i]}`, pt.x + 8, pt.y - 8);
    });
    if (pts.length === 4) {
      ctx.strokeStyle = "#bdb76b";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      ctx.lineTo(pts[1].x, pts[1].y);
      ctx.lineTo(pts[3].x, pts[3].y);
      ctx.lineTo(pts[2].x, pts[2].y);
      ctx.closePath();
      ctx.stroke();
    }
    const prompt = pts.length < 4
      ? `Click ${HOMO_LABELS[pts.length]} (${pts.length}/4)`
      : "4/4 selected. Enter LANE_WIDTH_M and REFERENCE_DISTANCE_M, then Save Homography.";
    setHud(prompt);
  }

  function drawPoly(points, color, closed) {
    if (!points.length) return;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    points.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
    if (points.length > 1) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      points.slice(1).forEach((pt) => ctx.lineTo(pt.x, pt.y));
      if (closed) ctx.closePath();
      ctx.stroke();
    }
  }

  function drawArrow(start, end, color) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    const ang = Math.atan2(end.y - start.y, end.x - start.x);
    ctx.beginPath();
    ctx.moveTo(end.x, end.y);
    ctx.lineTo(end.x - 12 * Math.cos(ang - 0.4), end.y - 12 * Math.sin(ang - 0.4));
    ctx.lineTo(end.x - 12 * Math.cos(ang + 0.4), end.y - 12 * Math.sin(ang + 0.4));
    ctx.closePath();
    ctx.fill();
  }

  function drawLanes() {
    state.finishedLanes.forEach((lane, i) => {
      const color = LANE_COLORS[i % LANE_COLORS.length];
      drawPoly(lane.polygon, color, true);
      const cx = lane.polygon.reduce((s, p) => s + p.x, 0) / lane.polygon.length;
      const cy = lane.polygon.reduce((s, p) => s + p.y, 0) / lane.polygon.length;
      const label = `Lane ${i + 1}` + (lane.heading != null ? `  ${lane.heading.toFixed(0)} deg` : "  no heading");
      ctx.font = "14px Segoe UI";
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 3;
      ctx.strokeText(label, cx, cy);
      ctx.fillStyle = color;
      ctx.fillText(label, cx, cy);
      if (lane.arrow) drawArrow(lane.arrow.start, lane.arrow.end, color);
    });
    if (state.pendingPolygon) {
      drawPoly(state.pendingPolygon, "#22d3ee", true);
      state.currentArrow.forEach((pt) => {
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });
      if (state.currentArrow.length === 2) {
        drawArrow(state.currentArrow[0], state.currentArrow[1], "#fff");
      }
    } else {
      drawPoly(state.currentPolygon, LANE_COLORS[state.finishedLanes.length % LANE_COLORS.length], false);
    }
    if (state.pendingPolygon) {
      setHud(`Click direction arrow: ${state.currentArrow.length}/2 points`);
      $("laneStatus").textContent = `Lane ${state.finishedLanes.length + 1} boundary done — click travel start, then end (or Enter to skip heading)`;
    } else {
      setHud(`Current lane points: ${state.currentPolygon.length}`);
      $("laneStatus").textContent = `Lanes finished: ${state.finishedLanes.length}   Current lane points: ${state.currentPolygon.length}`;
    }
  }

  function renderHomoTable() {
    const rows = $("homoPoints").querySelectorAll("tr");
    HOMO_LABELS.forEach((_, i) => {
      const pt = state.homoPoints[i];
      rows[i].children[1].textContent = pt ? pt.x : "—";
      rows[i].children[2].textContent = pt ? pt.y : "—";
    });
  }

  function renderLaneTable() {
    const body = $("laneRows");
    if (!state.finishedLanes.length) {
      body.innerHTML = `<tr><td colspan="3" class="why">No lanes finished</td></tr>`;
      return;
    }
    body.innerHTML = state.finishedLanes.map((lane, i) => {
      const heading = lane.heading == null ? "—" : `${lane.heading.toFixed(1)} deg`;
      return `<tr><td>${i + 1}</td><td>${lane.polygon.length}</td><td>${heading}</td></tr>`;
    }).join("");
  }

  function onCanvasClick(ev) {
    if (state.stage !== "homography" && state.stage !== "lanes") return;
    const pt = canvasPoint(ev);
    if (state.stage === "homography") {
      if (state.homoPoints.length >= 4) return;
      state.homoPoints.push(pt);
      log(`${HOMO_LABELS[state.homoPoints.length - 1]}: (${pt.x}, ${pt.y})`);
      renderHomoTable();
      draw();
      return;
    }
    if (state.stage === "lanes") {
      if (state.pendingPolygon) {
        if (state.currentArrow.length < 2) state.currentArrow.push(pt);
        if (state.currentArrow.length === 2) finalizePendingLane();
      } else {
        state.currentPolygon.push(pt);
      }
      draw();
    }
  }

  function resetHomo() {
    state.homoPoints = [];
    renderHomoTable();
    draw();
    log("Homography points reset.");
  }

  async function saveHomography() {
    if (state.homoPoints.length !== 4) {
      log("Need 4 points: near-left, near-right, far-left, far-right.");
      return;
    }
    const laneWidth = Number($("lane_width_m").value);
    const refDist = Number($("reference_distance_m").value);
    if (!(laneWidth > 0) || !(refDist > 0)) {
      log("LANE_WIDTH_M and REFERENCE_DISTANCE_M must be > 0.");
      return;
    }
    const res = await fetch(`/api/sources/${encodeURIComponent(state.selectedId)}/homography`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        src: state.homoPoints.map((p) => [p.x, p.y]),
        lane_width_m: laneWidth,
        reference_distance_m: refDist,
        frame_index: state.frameIndex,
      }),
    });
    const payload = await res.json();
    if (!res.ok) {
      log(`Homography save failed: ${payload.detail || res.statusText}`);
      return;
    }
    applyRecord(payload);
    setStatus("Homography saved — box lanes");
    log(`Homography saved. lane_width_m=${laneWidth}  reference_distance_m=${refDist}`);
    await loadSources();
    setStage("lanes");
  }

  function finishCurrentLane() {
    if (state.pendingPolygon) {
      finalizePendingLane();
      return;
    }
    if (state.currentPolygon.length < 3) {
      log(`Need at least 3 points to finish a lane, have ${state.currentPolygon.length}.`);
      return;
    }
    state.pendingPolygon = state.currentPolygon.slice();
    state.currentPolygon = [];
    state.currentArrow = [];
    log(`Lane boundary finished with ${state.pendingPolygon.length} points. Click 2-point travel arrow, or Enter to skip heading.`);
    draw();
  }

  function headingFromArrow(start, end) {
    const homo = state.record && state.record.homography;
    if (!homo || !homo.src) return null;
    try {
      const H = getPerspectiveTransform(
        homo.src,
        [
          [0, 0],
          [homo.lane_width_m, 0],
          [0, homo.reference_distance_m],
          [homo.lane_width_m, homo.reference_distance_m],
        ],
      );
      const a = perspectiveTransform(H, start.x, start.y);
      const b = perspectiveTransform(H, end.x, end.y);
      return Math.atan2(b[0] - a[0], b[1] - a[1]) * 180 / Math.PI;
    } catch {
      return null;
    }
  }

  function finalizePendingLane() {
    if (!state.pendingPolygon) return;
    const arrow = state.currentArrow.length === 2
      ? { start: state.currentArrow[0], end: state.currentArrow[1] }
      : null;
    const heading = arrow ? headingFromArrow(arrow.start, arrow.end) : null;
    state.finishedLanes.push({
      polygon: state.pendingPolygon,
      arrow,
      heading,
    });
    log(`Lane ${state.finishedLanes.length} saved, heading: ${heading == null ? "none" : heading.toFixed(1) + " deg"}`);
    state.pendingPolygon = null;
    state.currentArrow = [];
    renderLaneTable();
    draw();
  }

  function undoPoint() {
    if (state.pendingPolygon) {
      if (state.currentArrow.length) state.currentArrow.pop();
    } else if (state.currentPolygon.length) {
      state.currentPolygon.pop();
    }
    draw();
  }

  function undoLane() {
    if (!state.finishedLanes.length) return;
    const removed = state.finishedLanes.pop();
    log(`Removed a lane with ${removed.polygon.length} points. ${state.finishedLanes.length} remain.`);
    renderLaneTable();
    draw();
  }

  function resetCurrentLane() {
    if (state.pendingPolygon) {
      state.currentArrow = [];
      log("Direction arrow cleared.");
    } else {
      state.currentPolygon = [];
      log("Current lane points cleared.");
    }
    draw();
  }

  async function saveLanes() {
    if (!state.finishedLanes.length && state.currentPolygon.length >= 3) {
      finishCurrentLane();
      if (state.pendingPolygon) finalizePendingLane();
    }
    if (!state.finishedLanes.length) {
      log("No lanes finished — nothing to save.");
      return;
    }
    const res = await fetch(`/api/sources/${encodeURIComponent(state.selectedId)}/lanes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        frame_index: state.frameIndex,
        lanes: state.finishedLanes.map((lane) => ({
          polygon: lane.polygon.map((p) => [p.x, p.y]),
          arrow: lane.arrow ? [[lane.arrow.start.x, lane.arrow.start.y], [lane.arrow.end.x, lane.arrow.end.y]] : null,
        })),
      }),
    });
    const payload = await res.json();
    if (!res.ok) {
      log(`Lane save failed: ${payload.detail || res.statusText}`);
      return;
    }
    applyRecord(payload);
    setStatus("Lanes saved — set event thresholds");
    log(`Saved ${payload.lanes.length} lane(s). Next: event identification parameters.`);
    await loadSources();
    setStage("events");
  }

  function fillEventForm(record) {
    const events = { ...EVENT_DEFAULTS, ...((record && record.events) || {}) };
    const vMin = events.V_MIN_KMH ?? record?.processing?.v_min_kmh;
    EVENT_KEYS.forEach((key) => {
      const el = $("ev_" + key);
      if (!el) return;
      if (key === "V_MIN_KMH") el.value = vMin == null || vMin === "" ? "" : vMin;
      else el.value = events[key] ?? EVENT_DEFAULTS[key];
    });
    const limit = record && record.place && record.place.speed_limit_kmh;
    $("ev_SPEED_LIMIT_SHOW").textContent = limit == null ? "—" : String(limit);
  }

  function collectEvents() {
    const body = {};
    EVENT_KEYS.forEach((key) => {
      const el = $("ev_" + key);
      if (!el) return;
      const raw = el.value.trim();
      if (raw === "") {
        if (key === "V_MIN_KMH") body[key] = null;
        return;
      }
      body[key] = Number(raw);
    });
    return body;
  }

  async function saveEvents() {
    if (!state.selectedId) {
      log("No source — ingest and box lanes first.");
      return false;
    }
    const res = await fetch(`/api/sources/${encodeURIComponent(state.selectedId)}/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectEvents()),
    });
    const payload = await res.json();
    if (!res.ok) {
      log(`Event save failed: ${payload.detail || res.statusText}`);
      return false;
    }
    applyRecord(payload);
    fillEventForm(payload);
    setStatus("Event parameters saved");
    log("Event identification parameters saved.");
    await loadSources();
    return true;
  }

  function renderIncidents(incidents) {
    const body = $("incidentRows");
    if (!incidents || !incidents.length) {
      body.innerHTML = `<tr><td colspan="4" class="why">No incidents yet</td></tr>`;
      return;
    }
    body.innerHTML = incidents.map((inc) => (
      `<tr><td>${inc.type || "—"}</td><td>${inc.source || "—"}</td><td>${inc.track_id ?? "—"}</td><td>${inc.ts_ms ?? "—"}</td></tr>`
    )).join("");
  }

  async function loadIncidents() {
    if (!state.selectedId) return;
    const res = await fetch(`/api/sources/${encodeURIComponent(state.selectedId)}/identify`);
    if (!res.ok) return;
    const data = await res.json();
    renderIncidents(data.incidents || []);
    const run = data.run || {};
    if (run.status === "running") $("identifyStatus").textContent = "Running…";
    else if (run.status === "ok") $("identifyStatus").textContent = `Done. ${ (data.incidents || []).length } incident(s).`;
    else if (run.status === "failed") $("identifyStatus").textContent = "Last run failed — see the log.";
    else $("identifyStatus").textContent = "Not run yet.";
    if (data.log_tail) log(data.log_tail.split("\n").slice(-3).join(" | "));
  }

  async function pollIdentify() {
    if (!state.selectedId) return;
    const res = await fetch(`/api/sources/${encodeURIComponent(state.selectedId)}/identify`);
    if (!res.ok) return;
    const data = await res.json();
    renderIncidents(data.incidents || []);
    const run = data.run || {};
    if (run.status === "running") {
      $("identifyStatus").textContent = "Running event identification…";
      setTimeout(pollIdentify, 2000);
      return;
    }
    $("runIdentifyBtn").disabled = false;
    if (run.status === "ok") {
      $("identifyStatus").textContent = `Done. ${(data.incidents || []).length} incident(s).`;
      setStatus("Identification complete");
      log(`Identification complete: ${(data.incidents || []).length} incident(s).`);
      try {
        const srcRes = await fetch(`/api/sources/${encodeURIComponent(state.selectedId)}`);
        if (srcRes.ok) applyRecord(await srcRes.json());
      } catch { /* tab unlock is best-effort */ }
    } else {
      $("identifyStatus").textContent = "Identification failed — see the log dock.";
      setStatus("Identification failed");
      if (data.log_tail) log(data.log_tail);
    }
  }

  async function runIdentify() {
    const saved = await saveEvents();
    if (!saved) return;
    $("runIdentifyBtn").disabled = true;
    $("identifyStatus").textContent = "Starting…";
    const res = await fetch(`/api/sources/${encodeURIComponent(state.selectedId)}/identify`, { method: "POST" });
    const payload = await res.json();
    if (!res.ok) {
      $("runIdentifyBtn").disabled = false;
      log(`Identify failed to start: ${payload.detail || res.statusText}`);
      return;
    }
    applyRecord(payload);
    setStatus("Event identification running");
    log("Started objectdetection.py for this source.");
    pollIdentify();
  }

  function riskInputKey(sourceId) {
    return "catris.riskInput." + sourceId;
  }

  function numOrZero(el) {
    if (!el) return 0;
    const raw = String(el.value).trim();
    if (raw === "" || Number.isNaN(Number(raw))) return 0;
    return Number(raw);
  }

  function resetRiskForm() {
    RISK_SK_KEYS.forEach((key) => {
      const el = $("risk_sk_" + key);
      if (el) el.value = "0";
    });
    RISK_WEIGHT_KEYS.forEach((key) => {
      const el = $("risk_" + key);
      if (el) el.value = "0";
    });
    $("risk_window").value = "this_clip";
    state.riskMetrics = null;
    $("riskStatus").textContent = "Not computed yet.";
    $("riskHero").hidden = true;
    $("riskBreakdown").hidden = true;
  }

  function fillRiskForm(saved, riskRow) {
    resetRiskForm();
    const input = saved || {};
    const sK = input.s_k || {};
    RISK_SK_KEYS.forEach((key) => {
      const el = $("risk_sk_" + key);
      if (!el) return;
      el.value = sK[key] != null ? String(sK[key]) : "0";
    });
    RISK_WEIGHT_KEYS.forEach((key) => {
      const el = $("risk_" + key);
      if (!el) return;
      el.value = input[key] != null ? String(input[key]) : "0";
    });
    $("risk_window").value = (input.window || (riskRow && riskRow.window) || "this_clip");
  }

  function collectRisk() {
    const s_k = {};
    RISK_SK_KEYS.forEach((key) => {
      s_k[key] = numOrZero($("risk_sk_" + key));
    });
    return {
      s_k,
      a: numOrZero($("risk_a")),
      b: numOrZero($("risk_b")),
      c: numOrZero($("risk_c")),
      d: numOrZero($("risk_d")),
      window: ($("risk_window").value || "").trim() || "this_clip",
    };
  }

  function fmtRisk(value) {
    if (value == null || value === "") return "—";
    if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
    return String(value);
  }

  function liveWeights() {
    return {
      a: numOrZero($("risk_a")),
      b: numOrZero($("risk_b")),
      c: numOrZero($("risk_c")),
      d: numOrZero($("risk_d")),
    };
  }

  function combineLive(metrics, a, b, c, d) {
    const V = Number(metrics.V) || 0;
    const P = Number(metrics.P) || 0;
    const E = Number(metrics.E) || 0;
    const C = Number(metrics.C) || 0;
    const R = Math.max(0, Math.min(100, a * V + b * P + c * E + d * C));
    const band = R < 40 ? "cold" : R < 70 ? "warm" : "hot";
    const T = Number(metrics.T) || 1;
    const wK = metrics.w_k || {};
    const videoCounts = metrics.video_counts || {};
    const inferredCounts = metrics.inferred_counts || {};
    const contributions = {};
    RISK_SK_KEYS.forEach((k) => {
      contributions[k] = a * (wK[k] || 0) * (videoCounts[k] || 0) / T;
    });
    ["near_miss", "harsh_brake", "weave"].forEach((k) => {
      contributions[k] = b * (inferredCounts[k] || 0) / T;
    });
    const top_types = Object.keys(contributions)
      .sort((x, y) => contributions[y] - contributions[x])
      .slice(0, 2)
      .join(",");
    return {
      V, P, E, C, R, band, top_types,
      vehicle_count: metrics.vehicle_count,
      window: ($("risk_window").value || "").trim() || "this_clip",
    };
  }

  function renderRiskResult(row, statusText) {
    if (!row) {
      $("riskStatus").textContent = statusText || "Not computed yet.";
      $("riskHero").hidden = true;
      $("riskBreakdown").hidden = true;
      return;
    }
    $("riskStatus").textContent = statusText || "Live.";
    $("riskHero").hidden = false;
    $("riskBreakdown").hidden = false;
    $("riskR").textContent = fmtRisk(row.R);
    const band = row.band || "—";
    const tag = $("riskBand");
    tag.textContent = band;
    tag.className = "band-tag band-" + band;
    $("riskV").textContent = fmtRisk(row.V);
    $("riskP").textContent = fmtRisk(row.P);
    $("riskE").textContent = fmtRisk(row.E);
    $("riskC").textContent = fmtRisk(row.C);
    $("riskTop").textContent = row.top_types || "—";
    $("riskVehicles").textContent = fmtRisk(row.vehicle_count);
    $("riskWindowOut").textContent = row.window || "—";
  }

  function applyLiveRisk(statusText) {
    if (!state.riskMetrics) {
      renderRiskResult(null, statusText || "Waiting on metrics…");
      return;
    }
    const w = liveWeights();
    renderRiskResult(combineLive(state.riskMetrics, w.a, w.b, w.c, w.d), statusText);
  }

  async function fetchRiskMetrics() {
    if (!state.selectedId) return;
    const s_k = {};
    RISK_SK_KEYS.forEach((key) => {
      s_k[key] = numOrZero($("risk_sk_" + key));
    });
    const req = ++state.riskMetricsReq;
    $("riskStatus").textContent = "Fetching metrics…";
    const res = await fetch(`/api/sources/${encodeURIComponent(state.selectedId)}/risk/metrics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ s_k }),
    });
    const payload = await res.json().catch(() => ({ detail: res.statusText }));
    if (req !== state.riskMetricsReq) return;
    if (!res.ok) {
      state.riskMetrics = null;
      const detail = payload.detail || res.statusText;
      renderRiskResult(null, "Metrics failed.");
      log(`Risk metrics failed: ${detail}`);
      return;
    }
    state.riskMetrics = payload;
    applyLiveRisk("Live — Save to persist.");
  }

  function onSkChanged() {
    if (state.riskSkTimer) clearTimeout(state.riskSkTimer);
    state.riskSkTimer = setTimeout(() => {
      fetchRiskMetrics().catch((err) => log(String(err)));
    }, 350);
  }

  function onBlendChanged() {
    applyLiveRisk(state.riskMetrics ? "Live — Save to persist." : undefined);
  }

  async function loadRisk() {
    if (!state.selectedId) {
      resetRiskForm();
      return;
    }
    let saved = null;
    try {
      saved = JSON.parse(localStorage.getItem(riskInputKey(state.selectedId)) || "null");
    } catch {
      saved = null;
    }
    let row = state.record && state.record.risk ? state.record.risk : null;
    try {
      const res = await fetch(`/api/sources/${encodeURIComponent(state.selectedId)}/risk`);
      if (res.ok) {
        const data = await res.json();
        if (data && data.risk) row = data.risk;
      }
    } catch {
      /* show whatever we already have */
    }
    fillRiskForm(saved, row);
    if (row && !state.riskMetrics) renderRiskResult(row, "Saved score — fetching live metrics…");
    await fetchRiskMetrics();
  }

  async function saveRisk() {
    if (!state.selectedId) {
      log("No source — finish Events (run identification) first.");
      return;
    }
    const body = collectRisk();
    try {
      localStorage.setItem(riskInputKey(state.selectedId), JSON.stringify(body));
    } catch { /* ignore quota */ }
    $("saveRiskBtn").disabled = true;
    $("riskStatus").textContent = "Saving…";
    const res = await fetch(`/api/sources/${encodeURIComponent(state.selectedId)}/risk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await res.json().catch(() => ({ detail: res.statusText }));
    $("saveRiskBtn").disabled = false;
    if (!res.ok) {
      const detail = payload.detail || res.statusText;
      $("riskStatus").textContent = "Save failed.";
      setStatus("Risk save failed");
      log(`Risk save failed: ${detail}`);
      applyLiveRisk();
      return;
    }
    if (state.record) {
      state.record.risk = payload;
      state.record.stage = "risk";
      unlockLaterSteps(state.record);
    }
    renderRiskResult(payload, "Saved.");
    setStatus("Risk saved");
    log(`Risk saved R=${fmtRisk(payload.R)} band=${payload.band} window=${payload.window || body.window}`);
  }

  function linsolve(A, b) {
    const n = b.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let i = 0; i < n; i++) {
      let max = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(M[k][i]) > Math.abs(M[max][i])) max = k;
      }
      [M[i], M[max]] = [M[max], M[i]];
      const piv = M[i][i];
      if (Math.abs(piv) < 1e-12) throw new Error("singular");
      for (let j = i; j <= n; j++) M[i][j] /= piv;
      for (let k = 0; k < n; k++) {
        if (k === i) continue;
        const f = M[k][i];
        for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j];
      }
    }
    return M.map((row) => row[n]);
  }

  function getPerspectiveTransform(src, dst) {
    const A = [];
    const b = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i];
      const [X, Y] = dst[i];
      A.push([x, y, 1, 0, 0, 0, -X * x, -X * y]);
      b.push(X);
      A.push([0, 0, 0, x, y, 1, -Y * x, -Y * y]);
      b.push(Y);
    }
    const h = linsolve(A, b);
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  }

  function perspectiveTransform(H, x, y) {
    const w = H[6] * x + H[7] * y + H[8];
    return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
  }

  document.querySelectorAll(".menu-btn").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const menu = btn.parentElement;
      const already = menu.classList.contains("open");
      closeMenus();
      if (!already) menu.classList.add("open");
    });
  });
  document.addEventListener("click", closeMenus);

  document.querySelectorAll("[data-action]").forEach((el) => {
    el.addEventListener("click", () => {
      const action = el.dataset.action;
      closeMenus();
      if (action === "new") resetProject();
      if (action === "open") $("videoFile").click();
      if (action === "ingest") ingestSource();
      if (action === "about") $("aboutModal").hidden = false;
      if (action === "toggle-tree") {
        document.querySelector(".app").classList.toggle("hide-tree");
        el.classList.toggle("checked");
      }
      if (action === "toggle-log") {
        document.querySelector(".app").classList.toggle("hide-log");
        el.classList.toggle("checked");
      }
    });
  });

  document.querySelectorAll("[data-goto]").forEach((el) => {
    el.addEventListener("click", () => {
      closeMenus();
      if (el.disabled) return;
      if (el.dataset.goto === "homography" && !state.selectedId) return;
      if (el.dataset.goto === "lanes" && !(state.record && state.record.homography)) return;
      if (el.dataset.goto === "events" && !(state.record && state.record.lanes && state.record.lanes.length)) return;
      if (el.dataset.goto === "risk" && !eventsComplete(state.record)) return;
      if (el.dataset.goto === "ingest") {
        $("preview").hidden = !state.file && !state.selectedId;
        if (state.selectedId && !state.file) {
          $("preview").src = `/api/sources/${encodeURIComponent(state.selectedId)}/video`;
          $("preview").hidden = false;
          $("emptyView").hidden = true;
        }
      }
      setStage(el.dataset.goto);
    });
  });

  $("browseBtn").addEventListener("click", () => $("videoFile").click());
  $("aboutClose").addEventListener("click", () => { $("aboutModal").hidden = true; });
  $("aboutModal").addEventListener("click", (ev) => {
    if (ev.target === $("aboutModal")) $("aboutModal").hidden = true;
  });
  $("videoFile").addEventListener("change", () => {
    const file = $("videoFile").files && $("videoFile").files[0];
    if (file) {
      setStage("ingest");
      setPreviewFile(file);
    }
  });

  const dropZone = $("dropZone");
  dropZone.addEventListener("dragover", (ev) => {
    ev.preventDefault();
    dropZone.classList.add("drag");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag"));
  dropZone.addEventListener("drop", (ev) => {
    ev.preventDefault();
    dropZone.classList.remove("drag");
    const file = ev.dataTransfer.files && ev.dataTransfer.files[0];
    if (file) {
      setStage("ingest");
      setPreviewFile(file);
    }
  });

  $("ingestForm").addEventListener("submit", (ev) => {
    ev.preventDefault();
    ingestSource();
  });

  canvas.addEventListener("click", onCanvasClick);
  $("openFrameBtn").addEventListener("click", () => openCalibPopup());
  document.querySelectorAll("[data-open-frame]").forEach((el) => {
    el.addEventListener("click", () => openCalibPopup());
  });
  $("resetHomoBtn").addEventListener("click", resetHomo);
  $("saveHomoBtn").addEventListener("click", () => saveHomography().catch((err) => log(String(err))));
  $("undoPointBtn").addEventListener("click", undoPoint);
  $("finishLaneBtn").addEventListener("click", finishCurrentLane);
  $("undoLaneBtn").addEventListener("click", undoLane);
  $("saveLanesBtn").addEventListener("click", () => saveLanes().catch((err) => log(String(err))));
  $("saveEventsBtn").addEventListener("click", () => saveEvents().catch((err) => log(String(err))));
  $("runIdentifyBtn").addEventListener("click", () => runIdentify().catch((err) => log(String(err))));
  $("saveRiskBtn").addEventListener("click", () => saveRisk().catch((err) => log(String(err))));
  RISK_SK_KEYS.forEach((key) => {
    const el = $("risk_sk_" + key);
    if (el) el.addEventListener("input", onSkChanged);
  });
  RISK_WEIGHT_KEYS.forEach((key) => {
    const el = $("risk_" + key);
    if (el) el.addEventListener("input", onBlendChanged);
  });
  $("risk_window").addEventListener("input", () => {
    if (state.riskMetrics) applyLiveRisk("Live — Save to persist.");
  });
  $("prevFrameBtn").addEventListener("click", () => {
    if (state.frameIndex > 0) loadFrame(state.frameIndex - 1);
  });
  $("nextFrameBtn").addEventListener("click", () => {
    if (state.frameIndex + 1 < state.frameCount) loadFrame(state.frameIndex + 1);
  });

  function handleCalibKeys(ev) {
    if (ev.target && (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT" || ev.target.tagName === "TEXTAREA")) return;
    if (state.stage !== "homography" && state.stage !== "lanes") return;
    if (ev.key === "n") {
      ev.preventDefault();
      if (state.frameIndex + 1 < state.frameCount) loadFrame(state.frameIndex + 1);
    }
    if (ev.key === "p") {
      ev.preventDefault();
      if (state.frameIndex > 0) loadFrame(state.frameIndex - 1);
    }
    if (ev.key === "r") {
      ev.preventDefault();
      if (state.stage === "homography") resetHomo();
      else resetCurrentLane();
    }
    if (ev.key === "q" || ev.key === "Escape") {
      if (state.popup && !state.popup.closed) closeCalibPopup();
    }
    if (state.stage === "lanes") {
      if (ev.key === "u") { ev.preventDefault(); undoPoint(); }
      if (ev.key === "z") { ev.preventDefault(); undoLane(); }
      if (ev.key === "Enter" || ev.key === "c") { ev.preventDefault(); finishCurrentLane(); }
    }
  }

  document.addEventListener("keydown", (ev) => {
    if (ev.key === "Escape") {
      closeMenus();
      $("aboutModal").hidden = true;
    }
    if (ev.ctrlKey && ev.key.toLowerCase() === "o") {
      ev.preventDefault();
      $("videoFile").click();
    }
    if (ev.ctrlKey && ev.key.toLowerCase() === "n") {
      ev.preventDefault();
      resetProject();
    }
    if (ev.ctrlKey && ev.key === "Enter" && state.stage === "ingest") {
      ev.preventDefault();
      ingestSource();
    }
    if (typing()) return;
    handleCalibKeys(ev);
  });

  log("catris v0.1 — ingest → homography → lanes → events → risk.");
  loadSources().catch((err) => log(`Could not list sources: ${err}`));
})();
