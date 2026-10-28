// Agent views rendered from the Matterport skybox at the current viewpoint: a front view at the agent's
// heading and a 360° panorama strip centred on it, both with markers for navigable neighbours, the
// chosen next node, the best option and the goal direction. Continuous (VLN-CE) map nodes bring their own
// cube faces rendered in habitat (`sky`, see ce_runs.py), used the same way.
//
// Same projection as MatterSim: cube faces from the skybox images (+x=2, -x=4, +y=0, -y=5, +z=1, -z=3),
// direction = P^T Rz(pi + heading)^T Rx(elevation - pi/2)^T d, with d's y flipped because MatterSim
// returns GL rows bottom-up.

const VFOV = Math.PI / 3;          // R2R / MatterSim vertical field of view
const NS = 'http://www.w3.org/2000/svg';

const rotX = a => [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]];
const rotZ = a => [[Math.cos(a), -Math.sin(a), 0], [Math.sin(a), Math.cos(a), 0], [0, 0, 1]];
const tr = A => A[0].map((_, j) => A.map(row => row[j]));
const mul = (A, B) => A.map(row => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
const wrap = a => ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export const skyboxUrl = (scan, vp, i, sky) => sky ? `${sky}_${i}.jpg` : `/api/skybox/${scan}/${vp}/${i}.jpg`;
const loadImage = src => new Promise((ok, fail) => { const im = new Image(); im.onload = () => ok(im); im.onerror = fail; im.src = src; });

export class Views {
  constructor(el) {
    this.el = el;   // { front, frontSvg, pano, panoSvg, msg, panoMsg, hint, follow }
    this.cache = new Map();
    this.key = null;
    this.token = 0;
    this.look = { yaw: 0, pitch: 0, free: false };
    this.ctx = null;
    this.glCanvas = document.createElement('canvas');
    const gl = this.glCanvas.getContext('webgl', { preserveDrawingBuffer: true });
    if (!gl) { this.message('WebGL unavailable'); return; }
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, 'attribute vec2 p; varying vec2 ndc; void main(){ ndc = p; gl_Position = vec4(p, 0.0, 1.0); }'));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, `precision highp float; varying vec2 ndc;
      uniform samplerCube cube; uniform mat3 R; uniform vec2 tanv; uniform vec2 span; uniform int mode;
      void main() {
        vec3 d;
        if (mode == 0) d = vec3(ndc.x * tanv.x, -ndc.y * tanv.y, -1.0);
        else { float a = ndc.x * span.x, b = ndc.y * span.y; d = vec3(sin(a) * cos(b), -sin(b), -cos(a) * cos(b)); }
        gl_FragColor = textureCube(cube, R * d);
      }`));
    gl.linkProgram(prog);
    gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'p');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    this.gl = gl;
    this.u = Object.fromEntries(['R', 'tanv', 'span', 'mode'].map(n => [n, gl.getUniformLocation(prog, n)]));
    this.bindInteraction();
    new ResizeObserver(() => this.draw()).observe(el.pano);
    new ResizeObserver(() => this.draw()).observe(el.front);
  }

  // ctx = { scan, vp, rot (3x3 row-major), heading, elevation (the agent's camera), markers: [{rel, elev, kind,
  //         label, title}], sky (CE images), note: {text, title} (shown with the heading) }
  async show(ctx) {
    this.ctx = ctx;
    const token = ++this.token; // Invalidate pending loads even when returning to an already drawn view.
    if (!this.gl) return;
    if (!this.look.free) { this.look.yaw = 0; this.look.pitch = 0; }
    const key = ctx.sky || ctx.scan + ctx.vp;
    if (!ctx.missing && this.key === key) { this.message(''); return this.draw(); }
    // Never display the previous viewpoint under the new viewpoint's title while images load.
    this.key = null;
    this.clearCanvases();
    this.el.hint.textContent = this.el.hint.title = '';
    this.el.follow.hidden = true;
    if (ctx.missing) { this.message(ctx.missing); return; }
    let tex = this.cache.get(key);
    if (!tex) {
      this.message('Loading panorama…');
      try {
        const imgs = await Promise.all([0, 1, 2, 3, 4, 5].map(i => loadImage(skyboxUrl(ctx.scan, ctx.vp, i, ctx.sky))));
        if (token !== this.token) return;
        tex = this.upload(imgs);
        this.cache.set(key, tex);
        if (this.cache.size > 16) { const [k, v] = this.cache.entries().next().value; this.gl.deleteTexture(v); this.cache.delete(k); }
      } catch (e) {
        if (token === this.token) { this.message('No panorama for this viewpoint (skybox images not found)'); this.key = null; this.clearCanvases(); }
        return;
      }
    }
    if (token !== this.token) return;
    this.message('');
    this.key = key;
    this.gl.bindTexture(this.gl.TEXTURE_CUBE_MAP, tex);
    this.draw();
  }

  message(text) {
    this.el.msg.textContent = text;
    if (this.el.panoMsg) this.el.panoMsg.textContent = text;
  }

  upload(imgs) {
    const gl = this.gl, tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, tex);
    const faces = [[gl.TEXTURE_CUBE_MAP_POSITIVE_X, 2], [gl.TEXTURE_CUBE_MAP_NEGATIVE_X, 4], [gl.TEXTURE_CUBE_MAP_POSITIVE_Y, 0],
                   [gl.TEXTURE_CUBE_MAP_NEGATIVE_Y, 5], [gl.TEXTURE_CUBE_MAP_POSITIVE_Z, 1], [gl.TEXTURE_CUBE_MAP_NEGATIVE_Z, 3]];
    for (const [target, i] of faces) gl.texImage2D(target, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, imgs[i]);
    for (const [p, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]])
      gl.texParameteri(gl.TEXTURE_CUBE_MAP, p, v);
    return tex;
  }

  matrix(heading, elevation) {
    const r = this.ctx.rot;
    const P = [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]];
    const M = mul(mul(tr(P), tr(rotZ(Math.PI + heading))), tr(rotX(elevation - Math.PI / 2)));
    return new Float32Array([M[0][0], M[1][0], M[2][0], M[0][1], M[1][1], M[2][1], M[0][2], M[1][2], M[2][2]]);
  }

  render(canvas, mode, heading, elevation) {
    const gl = this.gl, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.round(canvas.clientWidth * dpr)), h = Math.max(2, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    this.glCanvas.width = w; this.glCanvas.height = h;
    gl.viewport(0, 0, w, h);
    gl.uniformMatrix3fv(this.u.R, false, this.matrix(heading, elevation));
    gl.uniform1i(this.u.mode, mode);
    const tv = Math.tan(VFOV / 2);
    gl.uniform2f(this.u.tanv, tv * w / h, tv);
    gl.uniform2f(this.u.span, Math.PI, Math.min(Math.PI / 2, Math.PI * h / w));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    canvas.getContext('2d').drawImage(this.glCanvas, 0, 0);
    return { w: canvas.clientWidth, h: canvas.clientHeight };
  }

  clearCanvases() {
    for (const c of [this.el.front, this.el.pano]) c.getContext('2d').clearRect(0, 0, c.width, c.height);
    this.el.frontSvg.innerHTML = this.el.panoSvg.innerHTML = '';
  }

  draw() {
    if (!this.gl || !this.key || !this.ctx) return;
    if (!this.el.front.clientWidth || !this.el.pano.clientWidth) return;   // hidden; the ResizeObserver redraws on show
    const c = this.ctx, yaw = this.look.yaw, pitch = (c.elevation || 0) + this.look.pitch, heading = c.heading + yaw;
    const F = this.render(this.el.front, 0, heading, pitch);
    const Pn = this.render(this.el.pano, 1, c.heading, 0);
    const spanY = Math.min(Math.PI / 2, Math.PI * Pn.h / Pn.w);
    // panorama overlay: heading ticks, front-view frame, markers
    const px = rel => (0.5 + wrap(rel) / (2 * Math.PI)) * Pn.w, py = el => (0.5 - el / (2 * spanY)) * Pn.h;
    let s = '';
    for (let d = -180; d < 180; d += 30) {
      const x = px(d * Math.PI / 180), name = { 0: 'front', 90: 'right', [-90]: 'left', [-180]: 'back' }[d];
      s += `<line class="v-tick" x1="${x}" x2="${x}" y1="${Pn.h - 6}" y2="${Pn.h}"/>`;
      if (name) s += `<text class="v-axis" x="${x + 3}" y="${Pn.h - 8}">${name}</text>`;
    }
    const hfov = 2 * Math.atan(Math.tan(VFOV / 2) * F.w / F.h);
    const fw = hfov / (2 * Math.PI) * Pn.w, fh = VFOV / (2 * spanY) * Pn.h;
    const fx = px(yaw) - fw / 2, fy = py(pitch) - fh / 2;
    s += `<rect class="v-frame" x="${fx}" y="${fy}" width="${fw}" height="${fh}"/>`;
    if (fx < 0) s += `<rect class="v-frame" x="${fx + Pn.w}" y="${fy}" width="${fw}" height="${fh}"/>`;
    if (fx + fw > Pn.w) s += `<rect class="v-frame" x="${fx - Pn.w}" y="${fy}" width="${fw}" height="${fh}"/>`;
    s += placeMarkers(c.markers.map(m => ({ m, x: px(m.rel), y: py(m.elev) })), Pn.w);
    this.el.panoSvg.setAttribute('viewBox', `0 0 ${Pn.w} ${Pn.h}`);
    this.el.panoSvg.innerHTML = s;
    // front overlay: markers projected with the same camera
    const tH = Math.tan(hfov / 2), tV = Math.tan(VFOV / 2);
    const front = [];
    for (const m of c.markers) {
      const a = wrap(m.rel - yaw), d = [Math.sin(a) * Math.cos(m.elev), Math.sin(m.elev), -Math.cos(a) * Math.cos(m.elev)];
      const fwd = [0, Math.sin(pitch), -Math.cos(pitch)], up = [0, Math.cos(pitch), Math.sin(pitch)];
      const z = d[0] * fwd[0] + d[1] * fwd[1] + d[2] * fwd[2];
      if (z < 0.05) continue;
      const x = (1 + d[0] / z / tH) / 2 * F.w, y = (1 - (d[0] * up[0] + d[1] * up[1] + d[2] * up[2]) / z / tV) / 2 * F.h;
      if (x < -20 || x > F.w + 20 || y < -20 || y > F.h + 20) continue;
      front.push({ m, x, y });
    }
    const f = placeMarkers(front, F.w);
    this.el.frontSvg.setAttribute('viewBox', `0 0 ${F.w} ${F.h}`);
    this.el.frontSvg.innerHTML = f;
    const deg = ((heading * 180 / Math.PI) % 360 + 360) % 360;
    const elev = Math.round(pitch * 180 / Math.PI), note = c.note || {};
    this.el.hint.textContent = `heading ${deg.toFixed(0)}°${elev ? ` · elevation ${elev}°` : ''}${this.look.free ? ' · free look' : ''}${note.text ? ` · ${note.text}` : ''}`;
    this.el.hint.title = note.title || '';
    this.el.follow.hidden = !this.look.free;
  }

  bindInteraction() {
    const front = this.el.front;
    let drag = null;
    front.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, yaw: this.look.yaw, pitch: this.look.pitch }; front.setPointerCapture(e.pointerId); });
    front.addEventListener('pointermove', e => {
      if (!drag) return;
      this.look.free = true;
      this.look.yaw = drag.yaw - (e.clientX - drag.x) * 0.006;
      this.look.pitch = Math.max(-1.3, Math.min(1.3, drag.pitch + (e.clientY - drag.y) * 0.006));
      this.draw();
    });
    front.addEventListener('pointerup', () => { drag = null; });
    // click on the panorama: look there in the front view
    this.el.pano.addEventListener('click', e => {
      if (!this.ctx) return;
      const r = this.el.pano.getBoundingClientRect(), spanY = Math.min(Math.PI / 2, Math.PI * r.height / r.width);
      this.look = { free: true, yaw: ((e.clientX - r.left) / r.width - 0.5) * 2 * Math.PI,
                    pitch: Math.max(-1.3, Math.min(1.3, (0.5 - (e.clientY - r.top) / r.height) * 2 * spanY)) - (this.ctx.elevation || 0) };
      this.draw();
    });
    this.el.follow.addEventListener('click', () => { this.look = { yaw: 0, pitch: 0, free: false }; this.draw(); });
  }

  prefetch(scan, vp, sky) { for (let i = 0; i < 6; i++) new Image().src = skyboxUrl(scan, vp, i, sky); }
}

function placeMarkers(items, width) {
  // Label rows: a label that would overlap one already placed moves up a row (with a leader line);
  // labels are kept inside the view horizontally.
  const placed = [];
  return items.map(({ m, x, y }) => {
    let ly = y - 16, lx = x;
    if (m.label) {
      const half = 3.3 * m.label.length + 4;
      lx = Math.min(Math.max(x, half), width - half);
      while (placed.some(p => Math.abs(p.ly - ly) < 13 && Math.abs(p.x - lx) < p.half + half)) ly -= 14;
      placed.push({ x: lx, ly, half });
    }
    return marker(m, x, y, ly, lx);
  }).join('');
}

function marker(m, x, y, ly = y - 16, lx = x) {
  const t = m.title ? `<title>${esc(m.title)}</title>` : '';
  const lead = m.label && ly < y - 20 ? `<line class="v-lead" x1="${x}" y1="${y - 9}" x2="${lx}" y2="${ly + 3}"/>` : '';
  const lbl = m.label ? `${lead}<text class="v-label" x="${lx}" y="${ly}" text-anchor="middle">${esc(m.label)}</text>` : '';
  switch (m.kind) {
    case 'goal':
      return `<g class="v-m">${t}<path class="v-goal" d="M${x},${y - 8} L${x + 8},${y} L${x},${y + 8} L${x - 8},${y}Z"/>${lbl}</g>`;
    case 'chosen':
      return `<g class="v-m">${t}<circle class="v-chosen" cx="${x}" cy="${y}" r="11"/><circle class="v-cand" cx="${x}" cy="${y}" r="5"/>${lbl}</g>`;
    case 'best':
      return `<g class="v-m">${t}<circle class="v-best" cx="${x}" cy="${y}" r="14"/>${lbl}</g>`;
    case 'remote':
      return `<g class="v-m">${t}<circle class="v-remote" cx="${x}" cy="${y}" r="10"/>${lbl}</g>`;
    case 'visited':
      return `<g class="v-m">${t}<circle class="v-visited" cx="${x}" cy="${y}" r="4"/></g>`;
    default:
      return `<g class="v-m">${t}<circle class="v-cand" cx="${x}" cy="${y}" r="6"/>${lbl}</g>`;
  }
}
