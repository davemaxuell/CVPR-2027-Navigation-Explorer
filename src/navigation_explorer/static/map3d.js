// 3D building view: the scan's reconstructed mesh (ceilings removed, with optional floor cutaway),
// the navigation graph on the floor, ground-truth / agent / compared paths, DUET's graph memory, and an
// agent marker that walks smoothly along each hop. Matterport coordinates are kept (z up, metres).
// The whole building loads first at low detail; then the floor being shown is replaced by its own
// high-detail mesh (5-10x more triangles, so the vertex colours are much sharper). On top of that, the
// panorama photos of the agent's current viewpoint are projected onto every surface that viewpoint sees.
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from './vendor/CSS2DRenderer.js';
import { successRadius } from './presentation.js';
import { skyboxUrl } from './views.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888';
const color = name => new THREE.Color(css(name));

export class Map3D {
  constructor(container, hooks = {}) {
    this.el = container;
    this.hooks = hooks;                     // { onPick(vp) }
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.localClippingEnabled = true;
    container.appendChild(this.renderer.domElement);
    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = 'labels3d';
    container.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = color('--surface-2');
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.05, 400);
    this.camera.position.set(8, -8, 10);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.addEventListener('start', () => { if (this.mode === 'chase') this.setCamera('orbit'); });
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x777777, 2.2));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(5, -8, 20);
    this.scene.add(sun);

    this.clip = new THREE.Plane(new THREE.Vector3(0, 0, -1), 100);
    this.world = new THREE.Group();        // building mesh (per scan)
    this.graph = new THREE.Group();        // navigation graph and room labels (per scan)
    this.static = new THREE.Group();       // per episode
    this.dynamic = new THREE.Group();      // per frame
    this.inspection = new THREE.Group();   // interface selection; independent of playback animation
    this.start = new THREE.Group();        // kept visible when navigation overlays are hidden
    this.goal = new THREE.Group();         // kept visible when navigation overlays are hidden
    this.scene.add(this.world, this.graph, this.static, this.dynamic, this.inspection, this.start, this.goal);
    this.agent = this.makeAgent();
    this.scene.add(this.agent);

    this.mode = 'chase';
    this.showNodes = true;
    this.presentation = { model: true, reference: true, comparison: true, detailed: false };
    this.frameState = null;
    this.cutaway = false;                  // show the full building height by default
    this.cut = 1.3;                         // metres above the shown floor
    this.floorMode = 'auto';
    this.meshes = new Map();
    this.details = new Map();               // `${scan}/f${floor}` -> high-detail floor mesh (null if missing)
    this.detailLoading = new Set();
    this.hd = true;
    this.baseBand = { value: new THREE.Vector3() };   // slab of the building mesh hidden by the detail mesh
    // Photo projection, shared by every building material: the skybox of the agent's viewpoint, and a cube
    // of distances from that viewpoint to the nearest surface (stored as 1/distance) for occlusion.
    this.photo = true;
    this.photoKey = null;
    this.skyboxes = new Map();              // `${scan}/${vp}` -> CubeTexture
    this.depthRT = new THREE.WebGLCubeRenderTarget(512, {
      type: THREE.HalfFloatType, generateMipmaps: false, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
    this.cubeCam = new THREE.CubeCamera(0.05, 200, this.depthRT);
    this.proj = { uVp: { value: new THREE.Vector3() }, uSkyRot: { value: new THREE.Matrix3() },
                  uSky: { value: null }, uDepth: { value: null }, uProj: { value: new THREE.Vector2() } };
    this.anim = null;
    this.agentPos = new THREE.Vector3();
    this.agentHeading = 0;
    this.camGoal = null;
    this.active = true;
    this.renderer.setAnimationLoop(() => this.tick());
    new ResizeObserver(() => this.resize()).observe(container);
    this.renderer.domElement.addEventListener('dblclick', () => this.setCamera('overview'));
  }

  // ---------------------------------------------------------------- materials & helpers
  mat(name, opts = {}) {
    return new THREE.MeshLambertMaterial({ color: color(name), clippingPlanes: [this.clip], ...opts });
  }
  floorZ(vp) { const n = this.scan.nodes[vp]; return n.pos[2] - n.h; }
  ground(vp, lift = 0.06) { const n = this.scan.nodes[vp]; return new THREE.Vector3(n.pos[0], n.pos[1], n.pos[2] - n.h + lift); }
  label(text, cls, pos) {
    const div = document.createElement('div');
    div.className = 'l3 ' + (cls || '');
    div.textContent = text;
    const o = new CSS2DObject(div);
    o.position.copy(pos);
    return o;
  }
  cylinder(a, b, radius, material) {
    const d = new THREE.Vector3().subVectors(b, a), len = d.length() || 1e-3;
    const m = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 1, 10, 1), material);
    m.scale.set(1, len, 1);
    m.position.copy(a).addScaledVector(d, 0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
    m.userData = { a: a.clone(), b: b.clone(), len };
    return m;
  }
  clear(group) {
    for (const o of [...group.children]) {
      group.remove(o);
      if (o.userData.keep) continue;          // cached building mesh
      o.traverse(x => { x.geometry?.dispose?.(); if (x.element) x.element.remove(); });
    }
  }

  makeAgent() {
    const g = new THREE.Group();
    const body = new THREE.MeshLambertMaterial({ color: color('--model') });
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.05, 24), body);
    base.rotation.x = Math.PI / 2;
    const arrow = new THREE.Mesh(new THREE.ConeGeometry(0.16, 0.5, 16), body);
    arrow.position.set(0, 0.38, 0.06);        // local +y = heading direction
    const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 1.4, 6), body);
    stem.rotation.x = Math.PI / 2;
    stem.position.z = 0.7;
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.1, 16, 12), body);
    eye.position.z = 1.45;
    const pulse = new THREE.Mesh(new THREE.RingGeometry(0.28, 0.34, 40),
      new THREE.MeshBasicMaterial({ color: color('--model'), transparent: true, side: THREE.DoubleSide }));
    pulse.position.z = 0.02;
    g.add(base, arrow, stem, eye, pulse);
    g.userData = { pulse, materials: [body, pulse.material] };
    g.visible = false;
    return g;
  }

  // ---------------------------------------------------------------- scan
  async setScan(scan) {
    if (this.scan === scan) return;
    if (this.scan && this.scan.scan === scan.scan && this.scan.asset_query === scan.asset_query && this.scan.capabilities?.mesh === scan.capabilities?.mesh) {
      // same building, another graph: continuous (VLN-CE) episodes each bring their own points
      this.scan = scan;
      this.adj = adjacency(scan.edges);
      this.photoKey = null;
      this.updatePhoto();
      await this.baseReady;
      if (this.scan === scan) this.drawGraph();
      return;
    }
    this.scan = scan;
    this.adj = adjacency(scan.edges);
    this.clear(this.world);
    this.clear(this.graph);
    this.detail = null;
    this.detailKey = null;
    this.currentFloor = null;               // set by the next frame; floor ids are per scan
    this.baseBand.value.set(0, 0, 0);
    this.photoKey = null;
    this.proj.uProj.value.x = 0;
    this.hooks.status?.(scan.capabilities?.mesh ? 'Loading 3D building…' : '3D trajectory · no building mesh supplied');
    let ready;
    this.baseReady = new Promise(r => { ready = r; });
    const assetKey = `${scan.scan}${scan.asset_query || ''}`;
    let mesh = scan.capabilities?.mesh ? this.meshes.get(assetKey) : null;
    if (mesh === undefined) {
      try {
        mesh = await loadMesh(scan.mesh_url || `/api/mesh/${encodeURIComponent(scan.scan)}${scan.asset_query || ''}`, () => buildingMaterial(this.clip, this.baseBand, this.proj));
      } catch (e) {
        mesh = null;
        this.hooks.status?.(`No 3D mesh for ${scan.scan} (${e.message}); showing the graph only`);
      }
      this.meshes.set(assetKey, mesh);
      evict(this.meshes, 3);          // keep GPU memory bounded
    }
    ready();
    if (this.scan !== scan) return;
    if (mesh) { this.world.add(mesh); this.hooks.status?.(''); this.depthDirty = true; }
    this.meshVisible(this.showMesh !== false);
    this.updatePhoto();
    this.drawGraph();
  }

  // Navigation graph on the floor, with room names. Continuous episodes: only the reference points, since
  // ETPNav's map (nodes, ghost waypoints) grows during the episode and is drawn per frame (drawMapNow).
  drawGraph() {
    const scan = this.scan;
    this.clear(this.graph);
    const pts = [];
    const shown = vp => !['sample', 'compare'].includes(scan.nodes[vp].kind) && (!scan.ce || scan.nodes[vp].kind === 'ref');
    for (const [a, b] of scan.edges) if (shown(a) && shown(b)) pts.push(this.ground(a, 0.04), this.ground(b, 0.04));
    const edges = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: color('--text-muted'), transparent: true, opacity: 0.55, clippingPlanes: [this.clip] }));
    const ids = Object.keys(scan.nodes).filter(shown);
    const m = new THREE.Matrix4(), q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
    const dots = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.07, 0.07, 0.03, 12),
      new THREE.MeshLambertMaterial({ color: color('--text-muted'), clippingPlanes: [this.clip] }), ids.length);
    ids.forEach((vp, i) => dots.setMatrixAt(i, m.compose(this.ground(vp, 0.03), q, new THREE.Vector3(1, 1, 1))));
    this.graph.add(edges, dots);
    // room names
    this.rooms = [];
    const groups = new Map();
    for (const vp of ids) {
      const n = scan.nodes[vp];
      if (n.rid == null || !n.region) continue;
      const g = groups.get(n.rid) || { label: n.region, pts: [], floor: n.floor };
      g.pts.push(this.ground(vp, 1.0)); groups.set(n.rid, g);
    }
    for (const g of groups.values()) {
      const c = g.pts.reduce((s, p) => s.add(p), new THREE.Vector3()).divideScalar(g.pts.length);
      const l = this.label(g.label, 'room', c);
      l.userData.floor = g.floor;
      l.visible = this.presentation.detailed && (this.shownFloor() == null || g.floor === this.shownFloor());
      this.rooms.push(l);
      this.graph.add(l);
    }
  }

  meshVisible(on) {
    this.showMesh = on;
    const mesh = this.scan && this.meshes.get(`${this.scan.scan}${this.scan.asset_query || ''}`);
    if (mesh) mesh.visible = on;
    if (this.detail) this.detail.visible = on;
  }

  setDetail(on) { this.hd = on; this.ensureDetail(); }

  // The floor being shown (or the agent's floor when all floors are) gets its high-detail mesh, which
  // covers a slab from just below that floor to the top of the cut-away; inside that slab the building
  // mesh is hidden, so the two never overlap.
  async ensureDetail() {
    const scan = this.scan?.scan, floor = this.shownFloor() ?? this.currentFloor;
    const query = this.scan?.asset_query || '';
    const key = this.hd && this.scan?.capabilities?.mesh && scan && floor != null ? `${scan}/f${floor}${query}` : null;
    if (key === this.detailKey) return;
    this.detailKey = key;
    this.showDetail(key && this.details.get(key));
    if (!key || this.details.has(key) || this.detailLoading.has(key)) return;
    this.detailLoading.add(key);
    await this.baseReady;                  // the whole building first, it is what the user sees at once
    if (this.detailKey === key) this.hooks.status?.('Loading high-detail floor…');
    let mesh = null;
    try {
      mesh = await loadMesh(`/api/mesh/${encodeURIComponent(scan)}/f${floor}${query}`,
        slab => buildingMaterial(this.clip, { value: new THREE.Vector3(slab[0], slab[1], 2) }, this.proj));
    } catch (e) {
      console.warn(`no high-detail mesh for ${key}: ${e.message}`);
    }
    this.detailLoading.delete(key);
    this.details.set(key, mesh);
    evict(this.details, 2, this.detail);
    if (this.detailKey === key) { this.showDetail(mesh); this.hooks.status?.(''); }
  }

  showDetail(mesh) {
    if (this.detail) this.world.remove(this.detail);
    this.detail = mesh || null;
    this.depthDirty = true;
    const slab = this.detail?.userData.slab;
    this.baseBand.value.set(slab ? slab[0] : 0, slab ? slab[1] : 0, slab ? 1 : 0);
    if (!this.detail) return;
    this.detail.visible = this.showMesh !== false;
    this.world.add(this.detail);
  }

  setPhoto(on) { this.photo = on; this.updatePhoto(); }

  // Paint the agent's viewpoint panorama onto the building. The skybox loads here; tick() then renders
  // the distance cube from the viewpoint and turns the projection on.
  async updatePhoto() {
    const vp = this.photoVp || this.vp;
    const scan = this.scan, n = this.photo && scan && vp ? scan.nodes[vp] : null;
    const key = scan?.capabilities?.images && n?.rot?.length === 9 && n.sky ? n.sky : null;
    if (key === this.photoKey) return;
    this.photoKey = key;
    if (!key) { this.proj.uProj.value.x = 0; return; }   // otherwise the last photo stays until this one loads
    let tex = this.skyboxes.get(key);
    if (!tex) {
      try {   // cube faces in MatterSim's order: +x=2, -x=4, +y=0, -y=5, +z=1, -z=3
        tex = await new THREE.CubeTextureLoader().loadAsync([2, 4, 0, 5, 1, 3].map(i => skyboxUrl(scan.scan, vp, i, n.sky)));
      } catch (e) {
        console.warn(`no panorama for ${key}`);
        return;
      }
      tex.colorSpace = THREE.SRGBColorSpace;
      this.skyboxes.set(key, tex);
      for (const [k, t] of this.skyboxes) {
        if (this.skyboxes.size <= 6) break;
        if (t !== this.proj.uSky.value && k !== key) { t.dispose(); this.skyboxes.delete(k); }
      }
    }
    if (this.photoKey !== key) return;
    const r = n.rot;
    this.proj.uSky.value = tex;
    this.proj.uVp.value.set(...n.pos);
    this.proj.uSkyRot.value.set(r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]);   // P^T, as in views.js
    this.depthDirty = true;
  }

  // Distance from the projecting viewpoint to the nearest building surface in every direction, so the
  // panorama is painted only where that viewpoint actually sees the surface.
  renderDepth() {
    const r = this.renderer, p = this.proj;
    const hidden = this.scene.children.filter(o => o !== this.world && !o.isLight && o.visible);
    hidden.forEach(o => { o.visible = false; });
    const background = this.scene.background, clipping = r.localClippingEnabled;
    const clear = r.getClearColor(new THREE.Color()), alpha = r.getClearAlpha();
    this.scene.background = null;
    r.localClippingEnabled = false;         // walls above the cut-away still block the view
    r.setClearColor(0x000000, 1);           // 1/distance = 0: nothing in that direction
    p.uDepth.value = null;                  // never sample the cube being drawn
    p.uProj.value.y = 1;
    this.cubeCam.position.copy(p.uVp.value);
    this.cubeCam.updateMatrixWorld();
    this.cubeCam.update(r, this.scene);
    p.uProj.value.set(1, 0);
    p.uDepth.value = this.depthRT.texture;
    r.setClearColor(clear, alpha);
    r.localClippingEnabled = clipping;
    this.scene.background = background;
    hidden.forEach(o => { o.visible = true; });
    this.depthDirty = false;
  }

  setNodesVisible(on) {
    this.showNodes = on;
    this.graph.visible = this.static.visible = this.dynamic.visible = this.inspection.visible = on;
    this.agent.visible = on && !!this.ep;
    // Keep the start and goal readable even behind walls or on another floor when overlays are hidden.
    for (const endpoint of [this.start, this.goal]) endpoint.traverse(object => {
      if (!object.material) return;
      object.material.clippingPlanes = on ? [this.clip] : [];
      object.material.depthTest = on;
      object.material.depthWrite = on;
      object.material.needsUpdate = true;
      object.renderOrder = on ? 0 : 10;
    });
  }

  // Only overlay visibility changes here. Agent position, animation, and photographs are untouched.
  setPresentation(options) {
    Object.assign(this.presentation, options);
    this.updateRouteVisibility();
    this.updateClip();
    if (this.frameState) this.drawFrameOverlays(this.frameState);
  }

  updateRouteVisibility() {
    for (const object of this.static.children) {
      const { route, frame, secondary } = object.userData;
      object.visible = (!route || this.presentation[route]) && (!secondary || this.presentation.detailed) &&
        (frame == null || frame <= (this.frameState?.frame ?? 0));
    }
  }

  setInspection(vp, divergence = null) {
    this.inspection.traverse(object => object.material?.dispose());
    this.clear(this.inspection);
    const mark = (id, name, label) => {
      if (!id || !this.scan?.nodes[id]) return;
      const ring = new THREE.Mesh(new THREE.TorusGeometry(.6, .06, 8, 40), this.mat(name, { depthTest: false, depthWrite: false }));
      ring.position.copy(this.ground(id, .35)); ring.renderOrder = 12;
      const branch = name === '--model' ? ' model-branch' : name === '--compare' ? ' comparison-branch' : '';
      this.inspection.add(ring, this.label(label, 'inspection' + branch, this.ground(id, 1.6)));
    };
    if (vp) mark(vp, '--focus', `${vp.slice(0, 6)}${this.scan.nodes[vp]?.region ? ' · ' + this.scan.nodes[vp].region : ''}`);
    if (divergence) {
      for (const [id, name, label] of [[divergence.model, '--model', 'Model branch'], [divergence.comparison, '--compare', 'Comparison branch']]) {
        mark(id, name, label);
        if (id && divergence.shared && this.scan.nodes[id] && this.scan.nodes[divergence.shared]) {
          this.inspection.add(this.cylinder(this.ground(divergence.shared, .45), this.ground(id, .45), .09, this.mat(name)));
        }
      }
      if (!divergence.model || !divergence.comparison) mark(divergence.shared, '--focus', !divergence.model ? 'Model route ends here' : 'Comparison route ends here');
    }
  }

  // ---------------------------------------------------------------- episode
  setEpisode(ep, other) {
    this.ep = ep;
    this.frameState = null;
    this.clear(this.static);
    this.clear(this.dynamic);
    this.setInspection(null);
    this.clear(this.start);
    this.clear(this.goal);
    const gtMat = this.mat('--gt', { transparent: true, opacity: 0.75 });
    this.gtHops = [];
    ep.gt_path.forEach((vp, i) => {
      if (i === 0) return;
      const h = this.cylinder(this.ground(ep.gt_path[i - 1], 0.1), this.ground(vp, 0.1), 0.045, gtMat);
      h.userData.route = 'reference';
      this.gtHops.push(h);
      this.static.add(h);
    });
    if (other) {
      const cm = this.mat('--compare', { transparent: true, opacity: 0.7 });
      other.path.forEach((vp, i) => {
        if (i) {
          const h = this.cylinder(this.ground(other.path[i - 1], 0.14), this.ground(vp, 0.14), 0.025, cm);
          h.userData.route = 'comparison'; this.static.add(h);
        }
      });
      const end = other.path[other.path.length - 1];
      const label = this.label(`${other.name} ${other.endLabel || 'end'}`, 'muted', this.ground(end, 0.6));
      label.userData = { route: 'comparison', secondary: true }; this.static.add(label);
    }
    // start flag
    const s = this.ground(ep.start, 0);
    const pole = this.cylinder(s, s.clone().add(new THREE.Vector3(0, 0, 1.3)), 0.02, this.mat('--text-primary'));
    const flagShape = new THREE.Shape([new THREE.Vector2(0, 0), new THREE.Vector2(0.45, -0.15), new THREE.Vector2(0, -0.3)]);
    const flag = new THREE.Mesh(new THREE.ShapeGeometry(flagShape), this.mat('--gt', { side: THREE.DoubleSide }));
    flag.rotation.x = Math.PI / 2;
    flag.position.copy(s).add(new THREE.Vector3(0, 0, 1.3));
    this.start.add(pole, flag, this.label('start', '', s.clone().add(new THREE.Vector3(0, 0, 1.5))));
    // Goal marker and optional supplied success-radius guide.
    if (ep.goal) {
      const gpos = this.ground(ep.goal, 0.08);
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.42, 0.05, 10, 40), this.mat('--goal'));
      ring.position.copy(gpos);
      const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 2.6, 12), this.mat('--goal', { transparent: true, opacity: 0.45 }));
      beam.rotation.x = Math.PI / 2;
      beam.position.copy(gpos).add(new THREE.Vector3(0, 0, 1.3));
      const zoneMat = this.mat('--goal', { transparent: true, opacity: 0.28, depthWrite: false });
      if (this.scan?.ce && successRadius(ep) != null) {
        const d = new THREE.Mesh(new THREE.CircleGeometry(successRadius(ep), 64), this.mat('--goal', { transparent: true, opacity: 0.16, depthWrite: false }));
        d.position.copy(this.ground(ep.goal, 0.05));
        d.userData.route = 'reference';
        this.static.add(d);
      }
      for (const vp of this.scan?.ce ? [] : ep.zone) {
        const d = new THREE.Mesh(new THREE.CircleGeometry(0.45, 28), zoneMat);
        d.position.copy(this.ground(vp, 0.05));
        d.userData.route = 'reference';
        this.static.add(d);
      }
      this.goal.add(ring, beam, this.label('goal', 'goal', gpos.clone().add(new THREE.Vector3(0, 0, 2.8))));
    }
    // agent hops (built once, shown up to the current frame)
    const kinds = { local: this.mat('--model'), remote: this.mat('--remote'), return: this.mat('--remote'), start: this.mat('--model') };
    const seen = new Map();
    this.hops = ep.frames.map((f, i) => {
      if (!i) return null;
      const a = ep.frames[i - 1].vp, b = f.vp, key = a < b ? a + b : b + a, n = seen.get(key) || 0;
      seen.set(key, n + 1);
      const pa = this.ground(a, 0.18 + n * 0.12), pb = this.ground(b, 0.18 + n * 0.12);
      const h = this.cylinder(pa, pb, f.kind === 'local' ? 0.06 : 0.045, kinds[f.kind]);
      Object.assign(h.userData, { route: 'model', frame: i });
      h.visible = false;
      this.static.add(h);
      return h;
    });
    this.setNodesVisible(this.showNodes);
    this.updateRouteVisibility();
    this.agentPos.copy(this.ground(ep.frames[0].vp, 0));
    this.agentHeading = ep.frames[0].heading;
    this.camGoal = null;
    this.snap = true;
    this.setCamera(this.mode, true);
  }

  // ---------------------------------------------------------------- frame
  // st = { frame, fr, dec, state (visited/frontier/scores), counts, chosen, oracle, lost, final, labels, subGoal, animate }
  setFrame(st) {
    const ep = this.ep;
    if (!ep) return;
    const f = st.frame;
    this.frameState = st;
    this.hops.forEach((h, i) => { if (h) { h.visible = this.presentation.model && i <= f; this.stretch(h, 1); } });
    const to = this.ground(st.fr.vp, 0);
    if (st.animate && f > 0 && this.hops[f]) {
      this.anim = { from: this.ground(ep.frames[f - 1].vp, 0), to, h0: this.agentHeading, h1: st.fr.heading,
                    t0: performance.now(), dur: st.animate, hop: this.hops[f] };
      this.stretch(this.hops[f], 0);
    } else {
      this.anim = null;
      this.agentPos.copy(to);
      this.agentHeading = st.fr.heading;
    }
    this.currentFloor = st.fr.floor;
    this.vp = st.fr.vp;
    this.photoVp = st.skyVp || st.fr.vp;     // continuous runs: the map node whose images are shown
    this.updatePhoto();

    this.drawFrameOverlays(st);
    this.setInspection(st.labels.selected, st.labels.divergence);
    this.updateClip();
    if (this.mode === 'overview' && !this.camGoal) this.setCamera('overview', true);
  }

  drawFrameOverlays(st) {
    const ep = this.ep;
    this.clear(this.dynamic);
    const vis = new THREE.SphereGeometry(0.11, 14, 10);
    const mats = { 1: this.mat('--model'), 2: this.mat('--warning'), 3: this.mat('--critical') };
    for (const vp of st.state.visited) {
      const c = Math.min(st.counts.get(vp) || 1, 3);
      const m = new THREE.Mesh(vis, mats[c]);
      m.position.copy(this.ground(vp, 0.3));
      this.dynamic.add(m);
      if (c >= 2) this.dynamic.add(this.label(`×${st.counts.get(vp)}`, c >= 3 ? 'crit' : 'warn', this.ground(vp, 0.65)));
      if (st.labels.stop && ep.stop_probs[vp] != null && ep.stop_probs[vp] >= 0.05)
        this.dynamic.add(this.label(`stop ${(ep.stop_probs[vp] * 100).toFixed(0)}%`, 'muted', this.ground(vp, 0.95)));
    }
    const box = new THREE.EdgesGeometry(new THREE.BoxGeometry(0.3, 0.3, 0.3));
    const boxMat = new THREE.LineBasicMaterial({ color: color('--text-secondary'), clippingPlanes: [this.clip] });
    if (this.scan.ce) this.drawMapNow(st);
    for (const vp of st.state.frontier) {
      if (st.counts.get(vp)) continue;
      const b = new THREE.LineSegments(box, boxMat);
      b.position.copy(this.ground(vp, 0.2));
      // continuous runs: a ghost waypoint's size follows the planner's score for it
      if (this.scan.ce && st.state.score) b.scale.setScalar(0.6 + 1.8 * Math.sqrt(st.state.score.get(vp) || 0));
      this.dynamic.add(b);
      if (st.labels.scores && st.top.has(vp) && (this.presentation.detailed || vp === st.chosen))
        this.dynamic.add(this.label(pct(st.state.score.get(vp)), 'score', this.ground(vp, 0.55)));
    }
    const ring = (vp, name, r) => {
      const m = new THREE.Mesh(new THREE.TorusGeometry(r, 0.035, 8, 40), this.mat(name));
      m.position.copy(this.ground(vp, 0.2));
      this.dynamic.add(m);
    };
    if (st.oracle && this.presentation.reference) ring(st.oracle, '--gt', 0.5);
    if (st.chosen) ring(st.chosen, '--model', 0.36);
    if (st.subGoal && this.presentation.reference) {
      ring(st.subGoal, '--gt', 0.28);
      if (this.presentation.detailed) this.dynamic.add(this.label(`sub-goal ${st.subIdx + 1}`, 'gt', this.ground(st.subGoal, 1.25)));
    }
    if (st.lost) this.dynamic.add(this.label('✕ lost', 'crit', this.ground(st.lost, 0.9)));
    if (st.final) this.dynamic.add(this.label(st.final.text, 'stop ' + st.final.cls, this.ground(ep.final, 1.0)));
    // the ground-truth chunk of the current sub-task is drawn brighter
    if (this.gtHops) this.gtHops.forEach((h, i) => { h.scale.x = h.scale.z = st.chunk && i + 1 > st.chunk[0] - 1 && i + 1 <= st.chunk[1] - 1 ? 1.9 : 1; });
  }

  // Continuous runs: ETPNav's map at this decision (edges between its nodes, fainter ones to its ghost
  // waypoints) and, at the decision point, the waypoint predictor's output as spokes on the floor.
  drawMapNow(st) {
    const known = new Set(st.state.visited), ghosts = new Set(st.state.frontier), solid = [], faint = [];
    for (const a of known) {
      for (const b of this.adj.get(a) || []) {
        if (ghosts.has(b)) faint.push(this.ground(a, 0.12), this.ground(b, 0.12));
        else if (known.has(b) && a < b) solid.push(this.ground(a, 0.12), this.ground(b, 0.12));
      }
    }
    const lines = (pts, name, opacity) => {
      if (pts.length) this.dynamic.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({ color: color(name), transparent: true, opacity, clippingPlanes: [this.clip] })));
    };
    lines(solid, '--model', 0.8);
    lines(faint, '--text-secondary', 0.6);
    if (!st.atDecisionNode || !st.dec.cands?.length) return;
    const from = this.ground(st.dec.at, 0.08), spokes = [];
    const dot = new THREE.CircleGeometry(0.07, 16), dotMat = this.mat('--text-secondary', { side: THREE.DoubleSide });
    for (const [p] of st.dec.cands) {
      const to = new THREE.Vector3(p[0], p[1], p[2] + 0.08);
      spokes.push(from, to);
      const d = new THREE.Mesh(dot, dotMat);
      d.position.copy(to);
      this.dynamic.add(d);
    }
    lines(spokes, '--text-secondary', 0.9);
  }

  stretch(h, t) {
    const { a, b, len } = h.userData, d = new THREE.Vector3().subVectors(b, a);
    h.scale.y = Math.max(len * t, 1e-3);
    h.position.copy(a).addScaledVector(d, t / 2);
  }

  // ---------------------------------------------------------------- floors, camera, loop
  setFloorMode(mode) { this.floorMode = mode; this.updateClip(); if (this.mode === 'overview') this.setCamera('overview', true); }
  setCutaway(on) { this.cutaway = on; this.updateClip(); }
  setCut(m) { this.cut = m; this.updateClip(); }
  shownFloor() { return this.floorMode === 'auto' ? this.currentFloor : this.floorMode === 'all' ? null : this.floorMode; }
  updateClip() {
    const fl = this.shownFloor();
    const floors = this.scan?.floors || [];
    const z = fl == null ? Math.max(...floors.map(f => f.z), 0) + this.cut
                         : (floors.find(f => f.id === fl)?.z ?? 0) + this.cut;
    // Without the cut-away a shown floor still hides the floors above it; the chase camera would sit in them.
    const own = floors.find(f => f.id === fl)?.z;
    const above = fl == null ? Infinity : Math.min(...floors.filter(f => f.z > own + 1).map(f => f.z));
    this.clip.constant = this.cutaway ? z : isFinite(above) ? above - 0.15 : 1e4;
    for (const l of this.rooms || []) l.visible = this.presentation.detailed && (fl == null || l.userData.floor === fl);
    this.ensureDetail();
  }

  setCamera(mode, instant = false) {
    if (mode === 'goal' && !this.ep?.goal) return;
    this.mode = mode;
    this.hooks.mode?.(mode);
    if (!this.ep) return;
    if (instant) this.snap = true;
    if (mode === 'goal') {
      const target = this.ground(this.ep.goal, 0.65);
      const position = target.clone().add(new THREE.Vector3(3.2, -4.2, 4.5));
      this.camGoal = { pos: position, target, instant };
      if (instant) {
        this.camera.position.copy(position);
        this.controls.target.copy(target);
        this.controls.update();
        this.camGoal = null;
      }
    } else if (mode === 'overview') {
      const fl = this.shownFloor();
      const vps = [...this.ep.path, ...this.ep.gt_path].filter(vp => fl == null || this.scan.nodes[vp].floor === fl);
      const box = new THREE.Box3();
      (vps.length ? vps : this.ep.path).forEach(vp => box.expandByPoint(this.ground(vp, 0)));
      const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
      const r = Math.max(size.x, size.y, 6) * 0.62 + 2;
      this.camGoal = { pos: c.clone().add(new THREE.Vector3(r * 0.35, -r * 0.75, r * 1.05)), target: c, instant };
    } else {
      this.camGoal = null;
    }
  }

  tick() {
    if (!this.active || !this.ep) return;
    const now = performance.now();
    if (this.anim) {
      const a = this.anim, t = Math.min(1, (now - a.t0) / a.dur), e = 1 - Math.pow(1 - t, 3);
      this.agentPos.lerpVectors(a.from, a.to, e);
      let dh = ((a.h1 - a.h0 + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
      this.agentHeading = a.h0 + dh * Math.min(1, e * 1.6);
      this.stretch(a.hop, e);
      if (t >= 1) { this.anim = null; this.agentHeading = a.h1; }
    }
    this.agent.position.copy(this.agentPos);
    this.agent.rotation.z = -this.agentHeading;          // heading is clockwise from +y
    const p = this.agent.userData.pulse, ph = (now % 1400) / 1400;
    p.scale.setScalar(1 + ph * 2.2);
    p.material.opacity = 0.9 * (1 - ph);

    const target = this.agentPos.clone().add(new THREE.Vector3(0, 0, 0.8));
    if (this.mode === 'chase') {
      const back = new THREE.Vector3(-Math.sin(this.agentHeading), -Math.cos(this.agentHeading), 0);
      const want = target.clone().addScaledVector(back, 4.2).add(new THREE.Vector3(0, 0, 3.6));
      this.camera.position.lerp(want, this.snap ? 1 : 0.08);
      this.controls.target.lerp(target, this.snap ? 1 : 0.15);
    } else if (this.mode === 'orbit') {
      const d = target.clone().sub(this.controls.target).multiplyScalar(this.snap ? 1 : 0.12);
      this.controls.target.add(d);
      this.camera.position.add(d);
    } else if (this.camGoal) {
      const g = this.camGoal, k = g.instant ? 1 : 0.1;
      this.camera.position.lerp(g.pos, k);
      this.controls.target.lerp(g.target, k);
      if (g.instant || this.camera.position.distanceTo(g.pos) < 0.05) this.camGoal = null;
    }
    this.snap = false;
    this.controls.update();
    if (this.depthDirty && this.photoKey && this.proj.uSky.value) this.renderDepth();
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  }

  resize() {
    const w = this.el.clientWidth, h = this.el.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.labels.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setActive(on) { this.active = on; if (on) this.resize(); }
  recolor() {   // theme switch: materials read CSS tokens when created; rebuild what depends on them
    if (!this.ep || !this.scan) return;
    const scan = this.scan; this.scan = null;
    this.agent.userData.materials.forEach(m => m.color = color('--model'));
    this.scene.background = color('--surface-2');
    return this.setScan(scan);
  }
}

const pct = x => (x == null ? '–' : (100 * x).toFixed(x >= 0.995 || x < 0.1 ? 1 : 0) + '%');

function adjacency(edges) {
  const adj = new Map();
  for (const [a, b] of edges) {
    (adj.get(a) || adj.set(a, new Set()).get(a)).add(b);
    (adj.get(b) || adj.set(b, new Set()).get(b)).add(a);
  }
  return adj;
}

// Keep at most n meshes of a cache (oldest first out), never the one on screen.
function evict(cache, n, keep) {
  for (const [k, m] of cache) {
    if (cache.size <= n) break;
    if (m && m === keep) continue;
    m?.geometry.dispose();
    m?.material.dispose();
    cache.delete(k);
  }
}

// Building mesh material: the reconstruction's vertex colours, clipped by the cut-away plane and by a
// z slab uBand = (lo, hi, mode): mode 1 hides the slab (building mesh under a floor's detail mesh),
// mode 2 shows only the slab (the detail mesh). `proj` (shared) adds the photo projection and its
// distance pass (uProj = (projection on, distance pass)).
function buildingMaterial(clip, band, proj) {
  const m = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, clippingPlanes: [clip] });
  m.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, proj, { uBand: band });
    sh.vertexShader = 'varying vec3 vWorldPos;\n' + sh.vertexShader.replace('#include <begin_vertex>',
      '#include <begin_vertex>\nvWorldPos = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    sh.fragmentShader = BUILDING_UNIFORMS + sh.fragmentShader
      .replace('#include <clipping_planes_fragment>', '#include <clipping_planes_fragment>\n' + BAND_DISCARD)
      .replace('#include <color_fragment>', '#include <color_fragment>\n' + PHOTO_COLOUR)
      .replace('#include <dithering_fragment>', '#include <dithering_fragment>\n' + DISTANCE_OUT);
  };
  return m;
}

const BUILDING_UNIFORMS = `varying vec3 vWorldPos;
uniform vec3 uBand;
uniform vec3 uVp;
uniform mat3 uSkyRot;
uniform samplerCube uSky;
uniform samplerCube uDepth;
uniform vec2 uProj;
`;
const BAND_DISCARD = 'if (uBand.z > 0.5 && ((vWorldPos.z > uBand.x && vWorldPos.z < uBand.y) == (uBand.z < 1.5))) discard;';
// The panorama's colour replaces the vertex colour where the viewpoint sees this point: not behind a nearer
// surface, not at a grazing angle, and within 14 m (the panorama has ~5 mm pixels at 3 m). The skybox
// direction is -P^T * ray, the same projection as the agent views (views.js); lighting applies as before.
const PHOTO_COLOUR = `if (uProj.x > 0.5 && uProj.y < 0.5) {
  vec3 ray = vWorldPos - uVp;
  float dist = length(ray);
  vec3 dir = ray / dist;
  float inv = textureCube(uDepth, dir).r;
  float seen = step(dist, (inv > 0.0 ? 1.0 / inv : 1e4) + 0.03 + 0.02 * dist);
  float facing = abs(dot(normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos))), dir));
  float w = seen * smoothstep(0.08, 0.25, facing) * (1.0 - smoothstep(10.0, 14.0, dist));
  diffuseColor.rgb = mix(diffuseColor.rgb, textureCube(uSky, -(uSkyRot * dir)).rgb, w);
}`;
const DISTANCE_OUT = 'if (uProj.y > 0.5) gl_FragColor = vec4(1.0 / max(length(vWorldPos - uVp), 0.01), 0.0, 0.0, 1.0);';

// Mesh files from build_meshes.py: 'VLNM' (whole building) or 'VLNS' (one floor slab, with its z range).
async function loadMesh(url, material) {
  const r = await fetch(url);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
  const buf = await r.arrayBuffer();
  const dv = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== 'VLNM' && magic !== 'VLNS') throw new Error('bad mesh file');
  const nv = dv.getUint32(4, true), nf = dv.getUint32(8, true);
  const lo = [0, 1, 2].map(i => dv.getFloat32(12 + 4 * i, true)), hi = [0, 1, 2].map(i => dv.getFloat32(24 + 4 * i, true));
  let off = 36;
  const slab = magic === 'VLNS' ? [dv.getFloat32(36, true), dv.getFloat32(40, true)] : null;
  if (slab) off += 8;
  const q = new Uint16Array(buf, off, nv * 3); off += nv * 6 + (nv % 2 ? 2 : 0);
  const rgba = new Uint8Array(buf, off, nv * 4); off += nv * 4;
  const idx = new Uint32Array(buf, off, nf * 3);
  const pos = new Float32Array(nv * 3), col = new Float32Array(nv * 3);
  const lut = new Float32Array(256).map((_, i) => Math.pow(i / 255, 2.2));   // sRGB -> linear
  for (let i = 0; i < nv; i++) {
    for (let k = 0; k < 3; k++) {
      pos[3 * i + k] = lo[k] + (q[3 * i + k] / 65535) * (hi[k] - lo[k]);
      col[3 * i + k] = lut[rgba[4 * i + k]];
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  g.computeBoundingSphere();
  // nothing reads the arrays after the upload (no picking on the building), so free the CPU copies
  for (const a of [g.index, ...Object.values(g.attributes)]) a.onUpload(function () { this.array = null; });
  const mesh = new THREE.Mesh(g, material(slab));
  mesh.userData = { keep: true, slab };
  return mesh;
}
