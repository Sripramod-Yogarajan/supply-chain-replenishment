// 3D network scene of the simulation dashboard (DECISIONS D-43, D-44).
// It draws the values handed over by Python (scene3d.scene_spec) and nothing else: no simulator
// rule runs here. Tower height = days of cover (capped), colour = cover status, a stockout is an
// empty outlined tower with a marker; roads are arcs; shipments sit on the arcs.
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/CSS2DRenderer.js";

const X = { supplier: 0, warehouse: 4.2, store: 8.4 };
const TOWER_H = 3.0; // scene units of a tower at the cap
const R_TOWER = 0.55;
const ARC_PEAK = 1.45;
const SUPPLIER_H = 0.6;
// Default cameras: front view for a wide frame; a corner view for a narrow one (expanded view,
// three or four policies), where the chain runs diagonally and uses the frame's height.
const HOMES = {
  wide: { pos: new THREE.Vector3(4.2, 4.9, 10.9), target: new THREE.Vector3(4.2, 1.2, -0.25) },
  narrow: { pos: new THREE.Vector3(14.5, 8.2, 11.5), target: new THREE.Vector3(4.7, 1.55, -0.4) },
};
const frameShape = () => (W / H >= 1.8 ? "wide" : "narrow");
let shape = null;

const root = document.getElementById("root");
const tip = document.getElementById("tip");
const titleEl = document.getElementById("title");
let W = Math.max(root.clientWidth, 1), H = Math.max(root.clientHeight, 1);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(W, H);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
root.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.setSize(W, H);
labels.domElement.className = "labels";
root.appendChild(labels.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, W / H, 0.1, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.minDistance = 5;
controls.maxDistance = 32;
controls.maxPolarAngle = Math.PI * 0.49;
controls.update();

scene.add(new THREE.HemisphereLight(0xffffff, 0xdfe4ea, 1.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.5);
sun.position.set(-4, 12, 8);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
Object.assign(sun.shadow.camera, { left: -4, right: 14, top: 8, bottom: -8, near: 1, far: 40 });
scene.add(sun);

let P = null; // palette
let built = null; // network layout, built once
let runId = null;
const hoverables = [];
const towers = {}; // node id -> parts and tween values
const dots = {}; // shipment key ("road|arrival day") -> parts and tween values
const roads = {}; // road id -> parts
let anim = null; // {t0, ms}
let dirty = true;

const Y = (days) => (days / built.cap) * TOWER_H;

// A label pinned to a 3D point. CSS2DRenderer sets the outer element's transform each frame, so
// the offset ("up": sits above the point, "right": starts at the point) is on an inner element.
function label(cls, where = "") {
  const outer = document.createElement("div");
  const inner = document.createElement("div");
  inner.className = cls + (where ? " " + where : "");
  outer.appendChild(inner);
  const obj = new CSS2DObject(outer);
  obj.inner = inner;
  scene.add(obj);
  return obj;
}

function layout(spec) {
  const pos = {};
  for (const typ of ["supplier", "warehouse", "store"]) {
    const g = spec.nodes.filter((n) => n.type === typ);
    g.forEach((n, i) => { // a tier's nodes step back and to the right, so labels do not stack
      const k = g.length === 1 ? 0 : i / (g.length - 1) - 0.5;
      pos[n.id] = new THREE.Vector3(X[typ] - 2.0 * k, 0, 3.0 * k);
    });
  }
  return pos;
}

function buildStatic(spec) {
  P = spec.palette;
  scene.background = new THREE.Color(P.panel);
  const pos = layout(spec);
  built = { pos, cap: spec.cap_days };
  for (const typ of ["supplier", "warehouse", "store"]) {
    const ps = spec.nodes.filter((n) => n.type === typ).map((n) => pos[n.id]);
    if (!ps.length) continue;
    const zs = ps.map((p) => p.z), xs = ps.map((p) => p.x);
    const lo = Math.min(...zs), hi = Math.max(...zs), depth = hi - lo + 2.0;
    const width = Math.max(...xs) - Math.min(...xs) + 2.0;
    const plate = new THREE.Mesh(
      new THREE.BoxGeometry(width, 0.12, depth),
      new THREE.MeshStandardMaterial({ color: P.page, roughness: 1 }),
    );
    plate.position.set((Math.max(...xs) + Math.min(...xs)) / 2, -0.06, (lo + hi) / 2);
    plate.receiveShadow = true;
    scene.add(plate);
    const t = label("tier");
    t.inner.textContent = typ === "store" ? "Stores" : typ === "supplier" ? "Supplier" : "Warehouse";
    t.position.set(X[typ], 0, (lo + hi) / 2 + depth / 2 + 0.3);
  }
  // ruler in days of cover on a back wall line, labels at the left end (clear of the stores)
  const rz = -2.7, x0 = -1.0, x1 = X.store + 2.0;
  const mat = new THREE.LineBasicMaterial({ color: P.grid });
  const capMat = new THREE.LineDashedMaterial({ color: P.faint, dashSize: 0.15, gapSize: 0.1 });
  for (const d of spec.ruler) {
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(x0, Y(d), rz), new THREE.Vector3(x1, Y(d), rz)]);
    const line = new THREE.Line(g, d === spec.cap_days ? capMat : mat);
    if (d === spec.cap_days) line.computeLineDistances();
    scene.add(line);
    const l = label("tick", "left");
    l.inner.textContent = d === spec.cap_days ? `cap ${d} d` : `${d} d`;
    l.position.set(x0 - 0.1, Y(d), rz);
  }
}

// Compile every kind of material once while the scene is built, so the first shipment, late
// shipment or stockout that appears later does not stall a day step on shader compilation.
function prewarm() {
  const geo = new THREE.BoxGeometry(1, 1, 1);
  const kinds = [
    new THREE.MeshStandardMaterial({ roughness: 0.5 }),
    new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false }),
    new THREE.MeshBasicMaterial({ color: P.ink }),
  ];
  const tmp = new THREE.Group();
  for (const m of kinds) { const x = new THREE.Mesh(geo, m); x.castShadow = true; tmp.add(x); }
  tmp.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: P.ink })));
  scene.add(tmp);
  renderer.compile(scene, camera);
  renderer.render(scene, camera); // also builds the shadow pass
  scene.remove(tmp);
}

function tower(n) {
  if (towers[n.id]) return towers[n.id];
  const pos = built.pos[n.id];
  const supplier = n.type === "supplier";
  const geo = supplier
    ? new THREE.BoxGeometry(1.0, 1, 1.0).translate(0, 0.5, 0)
    : new THREE.CylinderGeometry(R_TOWER, R_TOWER, 1, 48).translate(0, 0.5, 0);
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.04 }));
  mesh.castShadow = true;
  mesh.position.copy(pos);
  scene.add(mesh);
  // stockout: an empty tower one day of cover high, outlined, faintly tinted
  const ghost = new THREE.Group();
  const shell = new THREE.Mesh(
    new THREE.CylinderGeometry(R_TOWER, R_TOWER, 1, 48, 1, true).translate(0, 0.5, 0),
    new THREE.MeshBasicMaterial({ color: P.alert, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false }),
  );
  ghost.add(shell);
  for (const y of [0, 1]) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(R_TOWER, 0.028, 8, 56), new THREE.MeshBasicMaterial({ color: P.ink }));
    ring.rotation.x = Math.PI / 2;
    ring.position.y = y;
    ghost.add(ring);
  }
  for (const a of [0.25, 0.75, 1.25, 1.75]) { // four upright edges
    const e = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 1, 6).translate(0, 0.5, 0), new THREE.MeshBasicMaterial({ color: P.ink }));
    e.position.set(R_TOWER * Math.cos(a * Math.PI), 0, R_TOWER * Math.sin(a * Math.PI));
    ghost.add(e);
  }
  ghost.position.copy(pos);
  ghost.visible = false;
  scene.add(ghost);
  // cap marker: a dark cone on a tower that is taller than the cap
  const cone = new THREE.Mesh(new THREE.ConeGeometry(0.22, 0.32, 24), new THREE.MeshStandardMaterial({ color: P.ink }));
  cone.visible = false;
  scene.add(cone);
  // stockout marker: a red cone pointing down at the empty tower
  const flag = new THREE.Mesh(new THREE.ConeGeometry(0.2, 0.34, 24).rotateX(Math.PI), new THREE.MeshStandardMaterial({ color: P.alert }));
  flag.visible = false;
  scene.add(flag);
  const t = {
    mesh, ghost, cone, flag, supplier, pos, cur: 0, from: 0, target: 0,
    name: label("node", "up"),
  };
  mesh.userData.hover = "";
  shell.userData.hover = "";
  hoverables.push(mesh, shell);
  towers[n.id] = t;
  return t;
}

function arc(a, b) {
  const s = a.clone().setY(0.14), e = b.clone().setY(0.14);
  const mid = s.clone().lerp(e, 0.5);
  mid.y = ARC_PEAK;
  return new THREE.QuadraticBezierCurve3(s, mid, e);
}

function drawRoad(r) {
  let o = roads[r.id];
  if (!o) {
    const curve = arc(built.pos[r.from], built.pos[r.to]);
    o = roads[r.id] = { curve, group: new THREE.Group(), key: "" };
    scene.add(o.group);
    o.hit = new THREE.Mesh(new THREE.TubeGeometry(curve, 40, 0.2, 6), new THREE.MeshBasicMaterial({ visible: false }));
    scene.add(o.hit);
    hoverables.push(o.hit);
  }
  const radius = r.shipped ? 0.035 + 0.1 * Math.min(r.shipped / r.cap, 1) : 0.028;
  const colour = r.delayed ? P.alert : r.shipped ? P.edge : P.idle;
  const key = `${radius.toFixed(3)}|${colour}|${r.delayed}`;
  o.hit.userData.hover = r.hover;
  if (o.key === key) return;
  o.group.clear();
  const mat = new THREE.MeshStandardMaterial({ color: colour, roughness: 0.7 });
  if (r.delayed) { // dashed, as the 2D map's shocked road
    const n = 16;
    for (let i = 0; i < n; i++) {
      const pts = [];
      for (let k = 0; k <= 5; k++) pts.push(o.curve.getPoint((i + (0.6 * k) / 5) / n));
      o.group.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 5, radius, 10), mat));
    }
  } else {
    o.group.add(new THREE.Mesh(new THREE.TubeGeometry(o.curve, 64, radius, 12), mat));
  }
  o.group.children.forEach((m) => (m.castShadow = true));
  o.key = key;
}

function drawDots(spec) {
  const seen = new Set();
  for (const r of spec.roads) {
    for (const d of r.dots) {
      seen.add(d.key);
      let s = dots[d.key];
      if (!s || s.late !== d.late) {
        if (s) drop(d.key);
        const geo = d.late ? new THREE.BoxGeometry(1.6, 1.6, 1.6) : new THREE.SphereGeometry(1, 28, 18);
        const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: d.late ? P.late : P.dot, roughness: 0.45 }));
        if (d.late) mesh.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: P.ink })));
        mesh.castShadow = true;
        scene.add(mesh);
        hoverables.push(mesh);
        const start = anim === null && tweenOff ? d.f : 0.05;
        s = dots[d.key] = {
          mesh, curve: roads[r.id].curve, late: d.late, label: label(d.late ? "dot late" : "dot", "up"),
          f: start, from: start, target: d.f, size: 0, fromSize: 0, targetSize: d.size,
        };
      }
      s.from = s.f;
      s.target = d.f;
      s.fromSize = s.size;
      s.targetSize = d.size;
      s.dying = false;
      s.spec = d;
      s.mesh.userData.hover = d.hover;
      s.label.inner.textContent = String(d.remaining);
    }
  }
  for (const [k, s] of Object.entries(dots)) {
    if (!seen.has(k) && !s.dying) { // delivered (or gone): run to the receiver and shrink
      s.from = s.f; s.target = 1.0; s.fromSize = s.size; s.targetSize = 0; s.dying = true;
    }
  }
}

function drop(key) {
  const s = dots[key];
  scene.remove(s.mesh);
  scene.remove(s.label);
  hoverables.splice(hoverables.indexOf(s.mesh), 1);
  delete dots[key];
}

let tweenOff = false;
let current = null;

function render(spec) {
  if (!built) { buildStatic(spec); prewarm(); }
  const newRun = spec.run !== runId;
  if (newRun) for (const k of Object.keys(dots)) drop(k);
  runId = spec.run;
  const ms = newRun ? 0 : spec.transition_ms;
  tweenOff = ms <= 0;
  const chips = spec.roads.filter((r) => r.delayed)
    .map((r) => `<span class="chip">⚠ road ${r.id.replace("->", "→")} delayed ${r.delay_text}</span>`);
  for (const n of spec.nodes) for (const s of n.shocks) chips.push(`<span class="chip">⚠ ${n.id}: ${s}</span>`);
  titleEl.innerHTML = `<span style="color:${spec.colour}">●</span> ${spec.label}`
    + ` <span class="muted">· day ${spec.day}</span>` + chips.join("");
  for (const n of spec.nodes) {
    const t = tower(n);
    const out = n.stockout > 0;
    t.from = t.cur;
    t.target = t.supplier ? SUPPLIER_H : Math.max(Y(n.days), 0.0001);
    t.mesh.material.color.set(n.colour);
    t.mesh.visible = t.supplier || n.stock > 0;
    t.ghost.visible = out;
    t.ghost.scale.set(1, Y(1), 1);
    t.mesh.userData.hover = n.hover;
    t.ghost.children[0].userData.hover = n.hover;
    t.out = out;
    t.capped = n.capped;
    t.name.inner.innerHTML = t.supplier
      ? `<b>${n.id}</b><span class="muted"> · source</span>`
      : `<b>${n.id}</b> ${n.stock_text} u<br><span class="muted">${n.capped ? "▲ " : ""}${n.cover_text} d cover</span>`;
    if (out) t.name.inner.innerHTML += `<br><span class="mark">▼ stockout ${n.stockout} u</span>`;
  }
  for (const r of spec.roads) drawRoad(r);
  drawDots(spec);
  if (tweenOff) finish(); else anim = { t0: performance.now(), ms };
  current = spec;
  dirty = true;
}

function finish() { // jump to the end state at once
  anim = null;
  for (const t of Object.values(towers)) { t.cur = t.target; t.from = t.target; }
  for (const [k, s] of Object.entries(dots)) {
    if (s.dying) drop(k);
    else { s.f = s.target; s.from = s.target; s.size = s.targetSize; s.fromSize = s.targetSize; }
  }
  place(1);
}

const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);

function place(e) {
  for (const t of Object.values(towers)) {
    t.cur = t.from + (t.target - t.from) * e;
    t.mesh.scale.y = t.cur;
    const top = t.out ? Y(1) : t.cur;
    t.cone.visible = t.capped && !t.out;
    t.cone.position.set(t.pos.x, top + 0.2, t.pos.z);
    t.flag.visible = t.out;
    t.flag.position.set(t.pos.x, top + 0.3, t.pos.z);
    t.name.position.set(t.pos.x, top + (t.cone.visible ? 0.42 : t.out ? 0.52 : 0.12), t.pos.z);
  }
  for (const [k, s] of Object.entries(dots)) {
    s.f = s.from + (s.target - s.from) * e;
    s.size = s.fromSize + (s.targetSize - s.fromSize) * e;
    s.mesh.scale.setScalar(Math.max(s.size, 0.0001));
    const p = s.curve.getPoint(s.f);
    s.mesh.position.copy(p);
    s.label.position.set(p.x, p.y + s.size + 0.06, p.z);
    s.label.visible = !s.dying;
    if (s.dying && e >= 1) drop(k);
  }
}

function frame(now) {
  requestAnimationFrame(frame);
  if (anim) {
    const k = anim.ms > 0 ? Math.min((now - anim.t0) / anim.ms, 1) : 1;
    place(ease(k));
    if (k >= 1) anim = null;
    dirty = true;
  }
  if (controls.update()) dirty = true;
  if (!dirty) return;
  dirty = false;
  renderer.render(scene, camera);
  labels.render(scene, camera);
  declutter();
}
requestAnimationFrame(frame);

// Node labels never cover each other: where two overlap on screen, the upper one (further back)
// moves up until it is clear. Only the label's position changes, never its text.
function declutter() {
  const items = Object.values(towers).map((t) => t.name).filter((l) => l.visible)
    .map((l) => { l.inner.style.marginTop = "0px"; return l; })
    .map((l) => ({ l, r: l.inner.getBoundingClientRect() }))
    .sort((a, b) => b.r.top - a.r.top); // lowest on screen first
  const placed = [];
  for (const it of items) {
    let top = it.r.top, bottom = it.r.bottom;
    for (const p of placed) {
      const across = it.r.left < p.right - 2 && it.r.right > p.left + 2;
      if (across && bottom > p.top - 2 && top < p.bottom) {
        const up = bottom - p.top + 3;
        top -= up; bottom -= up;
      }
    }
    const dy = top - it.r.top;
    if (dy) it.l.inner.style.marginTop = `${2 * dy}px`; // the label is centred: 2 px per pixel
    placed.push({ left: it.r.left, right: it.r.right, top, bottom });
  }
}

// hover: the same text as the 2D map and the day table's values, with source tags
const ray = new THREE.Raycaster();
const ptr = new THREE.Vector2();
renderer.domElement.addEventListener("pointermove", (ev) => {
  const b = renderer.domElement.getBoundingClientRect();
  ptr.set(((ev.clientX - b.left) / b.width) * 2 - 1, -((ev.clientY - b.top) / b.height) * 2 + 1);
  ray.setFromCamera(ptr, camera);
  const shown = hoverables.filter((o) => o.visible !== false && (o.parent?.visible ?? true) && o.userData.hover !== undefined);
  const hit = ray.intersectObjects(shown, false).find((h) => h.object.userData.hover && (h.object.visible || h.object.material.visible === false));
  if (!hit) { tip.style.display = "none"; return; }
  tip.innerHTML = hit.object.userData.hover;
  tip.style.display = "block";
  const x = ev.clientX - b.left, y = ev.clientY - b.top;
  tip.style.left = Math.max(4, Math.min(x + 14, W - tip.offsetWidth - 4)) + "px";
  tip.style.top = (y - tip.offsetHeight - 12 > 4 ? y - tip.offsetHeight - 12 : y + 16) + "px";
});
renderer.domElement.addEventListener("pointerleave", () => (tip.style.display = "none"));
controls.addEventListener("change", () => (dirty = true));

function goHome() {
  const home = HOMES[shape];
  camera.position.copy(home.pos);
  controls.target.copy(home.target);
  controls.update();
  dirty = true;
}
document.getElementById("reset").addEventListener("click", goHome);

function resize() {
  W = Math.max(root.clientWidth, 1);
  H = Math.max(root.clientHeight, 1);
  renderer.setSize(W, H);
  labels.setSize(W, H);
  camera.aspect = W / H;
  root.classList.toggle("big", H >= 400); // the expanded view: larger labels for a projector
  root.classList.toggle("tiny", W < 420); // three or four small frames: day numbers on hover only
  const next = frameShape();
  // fit the network to the frame: by width in a wide frame, by the diagonal in a narrow one
  camera.zoom = next === "wide" ? Math.min(1.1, (W / H) / 2.3) : Math.max(0.45, Math.min(0.8, (W / H) / 1.3));
  camera.updateProjectionMatrix();
  if (next !== shape) { shape = next; goHome(); } // a new frame shape: its own default camera
  dirty = true;
}
new ResizeObserver(resize).observe(root);

// What is drawn, for the browser test: end-state values of every tower and shipment.
window.__drawn = () => ({
  day: current && current.day,
  moving: anim !== null,
  towers: Object.fromEntries(Object.entries(towers).map(([id, t]) => [id, {
    days: t.supplier ? null : (t.mesh.scale.y / TOWER_H) * built.cap,
    filled: t.mesh.visible, outline: t.ghost.visible, capped: t.cone.visible,
    colour: "#" + t.mesh.material.color.getHexString().toUpperCase(),
    marker: t.flag.visible, label: t.name.inner.textContent,
  }])),
  dots: Object.fromEntries(Object.entries(dots).map(([k, s]) => [k, {
    f: s.f, size: s.size, late: s.late, label: s.label.inner.textContent,
  }])),
});

// Where a tower is on screen (frame pixels), for the screenshots' hover view.
window.__where = (id) => {
  const t = towers[id];
  const v = t.pos.clone().setY(Math.max(t.cur, 0.3) / 2).project(camera);
  return { x: ((v.x + 1) / 2) * W, y: ((1 - v.y) / 2) * H };
};

// Streamlit component protocol (version 1), without the npm helper library
function send(type, extra) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...extra }, "*");
}
window.addEventListener("message", (ev) => {
  if (!ev.data || ev.data.type !== "streamlit:render") return;
  const spec = ev.data.args.spec;
  if (root.clientHeight !== spec.height) {
    root.style.height = spec.height + "px";
    resize();
  }
  render(spec);
  send("streamlit:setFrameHeight", { height: spec.height });
});
window.__render = render;
send("streamlit:componentReady", { apiVersion: 1 });
