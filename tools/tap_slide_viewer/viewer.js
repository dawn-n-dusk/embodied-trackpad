/* Camera controls never send robot commands. No remote assets or dependencies. */
const $ = id => document.getElementById(id);
let desired = null, presets = {}, generation = 0, acknowledged = 0, sending = false;
let streamId = null, state = null, stateReceived = 0;
let pointer = null, activeView = 'reset';
const views = Object.fromEntries(['overview', 'wrist'].map(name => [name, {
  canvas: $(name), key: null, received: 0, rendered: 0, failed: false, times: [],
}]));
const copy = value => JSON.parse(JSON.stringify(value));
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

async function request(url, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 3000);
  try {
    const response = await fetch(url, {...options, cache: 'no-store', signal: controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    // Consume inside the timeout, so a stalled response body also recovers.
    return url.includes('.png') ? {
      blob: await response.blob(), headers: response.headers,
    } : await response.json();
  } finally { clearTimeout(timer); }
}

function selectView(name) {
  activeView = name;
  document.querySelectorAll('[data-preset]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.preset === name));
  });
  $('viewLabel').textContent = {reset: '自由视角', top: '俯视', side: '侧视'}[name];
}

function changed() {
  if (!desired) return;
  desired.azimuth = (desired.azimuth % 360 + 360) % 360;
  desired.elevation = clamp(desired.elevation, -89, 89);
  desired.distance = clamp(desired.distance, 0.1, 2);
  generation++;
  sendCamera();
}

// One request in flight. Coalesce new input into desired without replacing it
// with an old reply or a rendered frame that is still catching up.
async function sendCamera() {
  if (sending || !desired || acknowledged === generation) return;
  sending = true;
  const sent = generation;
  const payload = copy(desired);
  try {
    const reply = await request('/camera.json', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    if (!reply.ok) throw new Error('Camera rejected');
    acknowledged = sent;
  } catch (_) { /* Keep the latest target; bounded retry also covers restarts. */ }
  finally {
    sending = false;
    if (acknowledged !== generation) setTimeout(sendCamera, 60);
  }
}

function preset(name) {
  if (!presets[name]) return;
  desired = copy(presets[name]);
  selectView(name);
  changed();
}
function zoom(factor) {
  if (!desired) return;
  desired.distance *= factor;
  changed();
}
const viewport = $('viewport');
viewport.addEventListener('dragstart', event => event.preventDefault());
viewport.addEventListener('contextmenu', event => event.preventDefault());
viewport.addEventListener('pointerdown', event => {
  if (event.button !== 0 || !event.isPrimary || !desired) return;
  event.preventDefault();
  viewport.focus({preventScroll: true});
  pointer = {id: event.pointerId, x: event.clientX, y: event.clientY};
  viewport.setPointerCapture(event.pointerId);
  viewport.classList.add('dragging');
});
viewport.addEventListener('pointermove', event => {
  if (!pointer || event.pointerId !== pointer.id) return;
  event.preventDefault();
  desired.azimuth -= (event.clientX - pointer.x) * 0.3;
  desired.elevation -= (event.clientY - pointer.y) * 0.3;
  pointer.x = event.clientX; pointer.y = event.clientY;
  selectView('reset');
  changed();
});
function endDrag(event) {
  if (!pointer || event.pointerId !== pointer.id) return;
  if (viewport.hasPointerCapture(pointer.id)) viewport.releasePointerCapture(pointer.id);
  pointer = null;
  viewport.classList.remove('dragging');
}
['pointerup', 'pointercancel', 'lostpointercapture'].forEach(name => viewport.addEventListener(name, endDrag));
viewport.addEventListener('wheel', event => {
  event.preventDefault();
  const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewport.clientHeight : 1;
  zoom(Math.exp(clamp(event.deltaY * unit, -160, 160) * 0.002));
}, {passive: false});
viewport.addEventListener('dblclick', () => preset('reset'));
viewport.addEventListener('keydown', event => {
  if (!desired) return;
  const step = 4;
  if (event.key === 'ArrowLeft') desired.azimuth += step;
  else if (event.key === 'ArrowRight') desired.azimuth -= step;
  else if (event.key === 'ArrowUp') desired.elevation += step;
  else if (event.key === 'ArrowDown') desired.elevation -= step;
  else if (event.key === '+' || event.key === '=') { event.preventDefault(); zoom(.88); return; }
  else if (event.key === '-') { event.preventDefault(); zoom(1.12); return; }
  else if (event.key.toLowerCase() === 'r') { preset('reset'); return; }
  else return;
  event.preventDefault(); selectView('reset'); changed();
});
document.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => preset(button.dataset.preset)));
$('zoomIn').onclick = () => zoom(.85);
$('zoomOut').onclick = () => zoom(1 / .85);
$('reset').onclick = () => preset('reset');
$('expand').onclick = () => {
  const expanded = document.body.classList.toggle('expanded');
  $('expand').setAttribute('aria-label', expanded ? '退出展开' : '展开场景');
  $('expand').title = expanded ? '退出展开' : '展开场景';
};

async function refreshFrame(name) {
  const view = views[name];
  const started = performance.now();
  try {
    const result = await request(`/${name}.png`);
    const key = result.headers.get('X-Stream-Id') + ':' + result.headers.get('X-Frame-Id');
    if (key !== view.key) {
      const bitmap = await createImageBitmap(result.blob);
      view.canvas.width = bitmap.width; view.canvas.height = bitmap.height;
      view.canvas.getContext('2d').drawImage(bitmap, 0, 0);
      bitmap.close();
      view.key = key;
      view.received = Date.now();
      view.rendered = Number(result.headers.get('X-Rendered-Wall')) * 1000;
      view.canvas.dataset.frameId = result.headers.get('X-Frame-Id');
      view.canvas.dataset.cameraRevision = result.headers.get('X-Camera-Revision');
      view.times.push(performance.now());
      view.times = view.times.slice(-24);
    }
    view.failed = false;
  } catch (_) { view.failed = true; }
  // Bounded sequential requests; no overlapping image decodes or object URLs.
  setTimeout(() => refreshFrame(name), Math.max(view.failed ? 800 : 80, 100 - (performance.now() - started)));
}

async function pollState() {
  try {
    const next = await request('/state.json');
    presets = next.camera_presets || {};
    if (!desired || next.stream_id !== streamId) {
      desired = copy(next.camera);
      streamId = next.stream_id;
      acknowledged = generation;
      selectView('reset');
    }
    state = next; stateReceived = Date.now();
    $('simTime').textContent = `仿真 ${(next.sim_time || 0).toFixed(1)} s`;
    const m = next.mount_info;
    const h = m?.reset_measurements?.palm_xpos_m?.[2];
    $('palmHeight').textContent = Number.isFinite(h) ? `${((h - m.pad_top_m) * 100).toFixed(1)} cm` : '—';
    $('clearance').textContent = m?.index_clearance_target_m != null ? `${(m.index_clearance_target_m * 1000).toFixed(1)} mm` : '—';
    $('lifetime').textContent = next.continuous ? '持续直播' : '限时预览';
    const cam = next.camera;
    if (cam) $('cameraReadout').textContent = `方位 ${cam.azimuth.toFixed(0)}°  /  俯仰 ${(-cam.elevation).toFixed(0)}°`;
  } catch (_) { stateReceived = 0; }
  setTimeout(pollState, 700);
}

function health() {
  const now = Date.now();
  const sourceLive = state?.status === 'LIVE' && now - stateReceived < 3000;
  const ready = {};
  for (const [name, view] of Object.entries(views)) {
    ready[name] = sourceLive && !view.failed && now - view.received < 2500 && now - view.rendered < 3000;
    $(name === 'overview' ? 'overviewState' : 'wristState').textContent = ready[name] ? '实时画面' : view.key ? '画面未更新' : '等待画面';
  }
  const live = ready.overview && ready.wrist;
  $('connection').textContent = live ? '直播中' : sourceLive ? '画面恢复中' : '正在重连';
  $('connection').classList.toggle('live', live);
  $('notice').hidden = live;
  $('notice').textContent = !sourceLive ? '连接中断，正在重连…' : !ready.overview ? '总览画面恢复中…' : '腕部画面恢复中…';
  const times = views.overview.times;
  const fps = live && times.length > 1 ? ((times.length - 1) * 1000 / (times.at(-1) - times[0])).toFixed(1) : '—';
  $('frameInfo').textContent = `${fps} fps  ·  帧 ${views.overview.canvas.dataset.frameId || '—'}`;
}
refreshFrame('overview'); refreshFrame('wrist'); pollState();
setInterval(health, 500);
