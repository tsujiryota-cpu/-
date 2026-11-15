// カメラ映像から牌を認識する (検出 FCN → 切り出し → 分類 CNN)。onnxruntime-web (WASM) で端末内で実行。
(function (root) {
  'use strict';
  const STR = 4, CW = 32, CH = 40, PAD = 1.08, JUNK = 34;
  const V = { ready: false, det: null, cls: null, cache: [], threads: 1 };

  V.init = async function (base) {
    const ort = root.ort;
    base = new URL(base || './', root.location.href).href;
    ort.env.wasm.wasmPaths = base + 'ort/';
    const iso = !!root.crossOriginIsolated;
    V.threads = iso ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1)) : 1;
    ort.env.wasm.numThreads = V.threads;
    const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    [V.det, V.cls] = await Promise.all([
      ort.InferenceSession.create(base + 'models/det.onnx', opt),
      ort.InferenceSession.create(base + 'models/cls.onnx', opt),
    ]);
    V.cDet = document.createElement('canvas');
    V.cFull = document.createElement('canvas');
    V.ready = true;
  };

  function decode(o, gw, gh, thr) {
    const n = gw * gh, out = [];
    for (let y = 0; y < gh; y++) {
      for (let x = 0; x < gw; x++) {
        const i = y * gw + x, h = o[i];
        if (h < thr) continue;
        let peak = true;
        for (let dy = -1; dy <= 1 && peak; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= gh) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const xx = x + dx;
            if (xx < 0 || xx >= gw || (!dx && !dy)) continue;
            const v = o[yy * gw + xx];
            if (v > h || (v === h && (dy < 0 || (dy === 0 && dx < 0)))) { peak = false; break; }
          }
        }
        if (!peak) continue;
        out.push({
          x: (x + 0.5 + o[n + i]) * STR, y: (y + 0.5 + o[2 * n + i]) * STR,
          w: 16 * Math.exp(o[3 * n + i]), h: 16 * Math.exp(o[4 * n + i]), score: h,
        });
      }
    }
    // 近すぎる重複を除去
    out.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const a of out) {
      let dup = false;
      for (const b of keep) {
        if (Math.abs(a.x - b.x) < Math.min(a.w, b.w) * 0.45 && Math.abs(a.y - b.y) < Math.min(a.h, b.h) * 0.45) { dup = true; break; }
      }
      if (!dup) keep.push(a);
    }
    return keep;
  }

  // 学習時の crop() と同じ幾何で 32x40 に切り出す。rot=1: 箱の右辺が上になるよう回す
  function sampleCrop(px, W, H, b, rot, dst, off) {
    const x0 = b.x - b.w * PAD / 2, y0 = b.y - b.h * PAD / 2, x1 = b.x + b.w * PAD / 2, y1 = b.y + b.h * PAD / 2;
    let ox, oy, ux, uy, vx, vy;
    if (!rot) { ox = x0; oy = y0; ux = (x1 - x0) / CW; uy = 0; vx = 0; vy = (y1 - y0) / CH; }
    else { ox = x1; oy = y0; ux = 0; uy = (y1 - y0) / CW; vx = (x0 - x1) / CH; vy = 0; }
    const step = Math.max(Math.abs(ux) + Math.abs(uy), Math.abs(vx) + Math.abs(vy));
    const sub = step > 1.5 ? [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]] : [[0, 0]];
    const plane = CW * CH, inv = 1 / (255 * sub.length);
    for (let v = 0; v < CH; v++) {
      for (let u = 0; u < CW; u++) {
        let r = 0, g = 0, bl = 0;
        for (const s of sub) {
          let sx = ox + (u + s[0]) * ux + (v + s[1]) * vx, sy = oy + (u + s[0]) * uy + (v + s[1]) * vy;
          sx = sx < 0 ? 0 : sx > W - 1.001 ? W - 1.001 : sx;
          sy = sy < 0 ? 0 : sy > H - 1.001 ? H - 1.001 : sy;
          const ix = sx | 0, iy = sy | 0, fx = sx - ix, fy = sy - iy;
          const p = (iy * W + ix) * 4, q = p + W * 4;
          const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
          r += px[p] * w00 + px[p + 4] * w10 + px[q] * w01 + px[q + 4] * w11;
          g += px[p + 1] * w00 + px[p + 5] * w10 + px[q + 1] * w01 + px[q + 5] * w11;
          bl += px[p + 2] * w00 + px[p + 6] * w10 + px[q + 2] * w01 + px[q + 6] * w11;
        }
        const i = off + v * CW + u;
        dst[i] = r * inv; dst[i + plane] = g * inv; dst[i + 2 * plane] = bl * inv;
      }
    }
  }

  // src: video / image / canvas。opt: {detW, thr, isHand(box)->bool, cacheMs}
  V.process = async function (src, opt) {
    opt = opt || {};
    const ort = root.ort;
    const vw = src.videoWidth || src.naturalWidth || src.width, vh = src.videoHeight || src.naturalHeight || src.height;
    if (!vw || !vh) return null;
    const t0 = performance.now();
    const dw = opt.detW || 640, dh = Math.max(16, Math.round(dw * vh / vw / 16) * 16);
    V.cDet.width = dw; V.cDet.height = dh;
    const dctx = V.cDet.getContext('2d', { willReadFrequently: true });
    dctx.drawImage(src, 0, 0, dw, dh);
    const dpx = dctx.getImageData(0, 0, dw, dh).data;
    const n = dw * dh, inp = new Float32Array(3 * n);
    for (let i = 0, j = 0; i < n; i++, j += 4) { inp[i] = dpx[j] / 255; inp[n + i] = dpx[j + 1] / 255; inp[2 * n + i] = dpx[j + 2] / 255; }
    const res = await V.det.run({ image: new ort.Tensor('float32', inp, [1, 3, dh, dw]) });
    const o = res.out.data, gw = dw / STR, gh = dh / STR;
    const dets = decode(o, gw, gh, opt.thr || 0.3);
    const t1 = performance.now();

    // 分類用のフル解像度フレーム (幅 1280 まで)
    const fw = Math.min(vw, 1280), fh = Math.round(fw * vh / vw);
    V.cFull.width = fw; V.cFull.height = fh;
    const fctx = V.cFull.getContext('2d', { willReadFrequently: true });
    fctx.drawImage(src, 0, 0, fw, fh);
    const fpx = fctx.getImageData(0, 0, fw, fh).data;
    const sx = fw / dw, sy = fh / dh;
    const boxes = dets.map((d) => ({ x: d.x * sx, y: d.y * sy, w: d.w * sx, h: d.h * sy, score: d.score }));
    if (opt.fillHand !== false) for (const f of V.handFill(boxes)) boxes.push(f);

    // 手牌以外は位置キャッシュを使って再分類を間引く
    const now = performance.now(), cacheMs = opt.cacheMs === undefined ? 900 : opt.cacheMs;
    const handFlags = opt.handOf ? opt.handOf(boxes) : boxes.map(() => true);
    const jobs = []; // {bi, rot}
    boxes.forEach((b, bi) => {
      b.hand = handFlags[bi];
      if (!b.hand && cacheMs > 0) {
        for (const c of V.cache) {
          if (now - c.time < cacheMs && Math.abs(c.x - b.x) < b.w * 0.2 && Math.abs(c.y - b.y) < b.h * 0.2 && Math.abs(c.w - b.w) < b.w * 0.2) {
            b.t = c.t; b.conf = c.conf; b.side = c.side; b.cached = c.time; return;
          }
        }
      }
      const a = b.w / b.h;
      if (a < 0.9) jobs.push({ bi, rot: 0 });
      else if (a > 1.12) jobs.push({ bi, rot: 1 });
      else { jobs.push({ bi, rot: 0 }); jobs.push({ bi, rot: 1 }); }
    });
    if (jobs.length) {
      const plane = 3 * CW * CH, buf = new Float32Array(jobs.length * plane);
      jobs.forEach((j, k) => sampleCrop(fpx, fw, fh, boxes[j.bi], j.rot, buf, k * plane));
      const r = await V.cls.run({ crops: new ort.Tensor('float32', buf, [jobs.length, 3, CH, CW]) });
      const p = r.prob.data;
      jobs.forEach((j, k) => {
        let bt = 0, bp = -1;
        for (let c = 0; c < 34; c++) { const v = p[k * 35 + c]; if (v > bp) { bp = v; bt = c; } }
        const b = boxes[j.bi];
        if (b.conf === undefined || bp > b.conf) { b.t = bt; b.conf = bp; b.side = j.rot === 1; b.junk = p[k * 35 + JUNK]; }
      });
    }
    V.cache = boxes.filter((b) => !b.hand && b.conf !== undefined).map((b) => ({ x: b.x, y: b.y, w: b.w, t: b.t, conf: b.conf, side: b.side, time: b.cached || now }));
    const t2 = performance.now();
    return { tiles: boxes, fw, fh, ms: { det: t1 - t0, cls: t2 - t1 }, nCls: jobs.length };
  };

  // ---------- 配置の解釈 (手牌 / 各家の河 / その他) ----------
  // 手牌候補: 同じ高さに並ぶ同サイズの正立牌の最大クラスタ。幾何だけで決める (分類前に使える)
  V.rowCluster = function (boxes) {
    const ok = (b) => b.w <= b.h * 0.95 && !(b.conf !== undefined && (b.junk > 0.9 || (b.fill && (b.junk > 0.5 || b.conf < 0.5))));
    let best = null, bestScore = 0;
    for (const s of boxes) {
      if (!ok(s) || s.fill) continue;
      const mem = [];
      for (const b of boxes) {
        if (!ok(b)) continue;
        if (Math.abs(b.y - s.y) < s.h * 0.4 && b.h > s.h * 0.8 && b.h < s.h * 1.25) mem.push(b);
      }
      const score = mem.filter((b) => !b.fill).length * s.w * s.w;
      if (score > bestScore) { bestScore = score; best = mem; }
    }
    if (!best) return [];
    return best.sort((a, b) => a.x - b.x);
  };
  // 手牌の列で、牌 1〜2 枚ぶんの隙間があれば検出漏れとみなして補完候補の箱を返す (分類で非牌なら捨てる)
  V.handFill = function (boxes) {
    const row = V.rowCluster(boxes), out = [];
    if (row.length < 2) return out;
    const ds = [];
    for (let i = 1; i < row.length; i++) { const d = row[i].x - row[i - 1].x; if (d < row[i].w * 1.3) ds.push(d); }
    ds.sort((a, b) => a - b);
    const ws = row.map((b) => b.w).sort((a, b) => a - b);
    const pitch = ds.length ? ds[ds.length >> 1] : ws[ws.length >> 1] * 1.02;
    for (let i = 1; i < row.length; i++) {
      const a = row[i - 1], b = row[i], r = (b.x - a.x) / pitch, k = Math.round(r);
      if ((k === 2 || k === 3) && Math.abs(r - k) < 0.3) {
        for (let j = 1; j < k; j++) {
          const f = j / k;
          out.push({ x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f, w: (a.w + b.w) / 2, h: (a.h + b.h) / 2, score: 0, fill: true });
        }
      }
    }
    return out;
  };
  V.findHand = function (boxes) {
    const best = V.rowCluster(boxes);
    if (!best.length) return [];
    // 大きな隙間(牌幅の0.75倍以上)で区切り、最長の区間を手牌とする → 右側の区間は自分の副露
    const segs = [[best[0]]];
    for (let i = 1; i < best.length; i++) {
      const a = best[i - 1], b = best[i];
      const gap = (b.x - b.w / 2) - (a.x + a.w / 2);
      if (gap > 0.75 * a.w) segs.push([b]); else segs[segs.length - 1].push(b);
    }
    segs.sort((a, b) => b.length - a.length || a[0].x - b[0].x);
    return segs[0];
  };
  V.handFlags = function (boxes) {
    const h = new Set(V.findHand(boxes));
    return boxes.map((b) => h.has(b));
  };

  // tiles: 分類済み (t, conf, junk)。戻り値 {hand:[...], rivers:[[自分],[下家(右)],[対面],[上家(左)]], others:[...], center}
  // 河は卓の中心を囲む風車状の配置: 自分と対面の河は正立/倒立(縦長)、下家と上家の河は横倒し(横長)。
  const med = (a) => { a = a.slice().sort((x, y) => x - y); return a.length ? a[a.length >> 1] : null; };
  V.layout = function (tiles, opt) {
    opt = opt || {};
    const minConf = opt.minConf || 0.45;
    const hand = V.findHand(tiles);
    const hs = new Set(hand);
    for (const b of tiles) b.zone = undefined;
    for (const b of hand) b.zone = 'hand';
    const rest = tiles.filter((b) => !hs.has(b) && !b.fill && b.conf >= minConf && !(b.junk > 0.5));
    const rivers = [[], [], [], []], others = [];
    if (!rest.length) return { hand, rivers, others, dora: [], center: null };
    const hw = hand.length ? med(hand.map((b) => b.w)) : Infinity;
    // 河の牌サイズ s (手牌より小さい牌の短辺の中央値)
    const small = rest.filter((b) => Math.min(b.w, b.h) < hw * 0.92);
    const s = med((small.length >= 3 ? small : rest).map((b) => Math.min(b.w, b.h)));
    const cand = rest.filter((b) => { const m = Math.min(b.w, b.h); return m > s * 0.6 && m < s * 1.5; });
    const isSide = (b) => b.w > b.h;
    const U = cand.filter((b) => !isSide(b)), Sd = cand.filter(isSide);
    const hx = hand.length ? (hand[0].x + hand[hand.length - 1].x) / 2 : null;
    const hy = hand.length ? med(hand.map((b) => b.y)) : null, hh = hand.length ? med(hand.map((b) => b.h)) : 0;
    // 中心の初期値
    let cx = U.length >= 3 ? med(U.map((b) => b.x)) : Sd.length >= 4 ? (Math.min(...Sd.map((b) => b.x)) + Math.max(...Sd.map((b) => b.x))) / 2 : hx;
    let cy = Sd.length >= 3 ? med(Sd.map((b) => b.y)) : hy !== null ? hy - 3.6 * hh : U.length ? (Math.min(...U.map((b) => b.y)) + Math.max(...U.map((b) => b.y))) / 2 : null;
    if (cx === null || cy === null) { for (const b of rest) { b.zone = 'other'; others.push(b); } return { hand, rivers, others, dora: [], center: null }; }
    // 河の範囲 (牌幅 s 単位, 各家から見た座標): 奥行き v は中心から R0〜R1、横 u は 6 枚幅 (±3) + 最終段のはみ出し分
    const R0 = opt.rMin || 1.6, R1 = opt.rMax || 9, UL = opt.uLeft || 4, UR = opt.uRight || 7.5;
    const assign = () => {
      for (const b of cand) {
        const dx = (b.x - cx) / s, dy = (b.y - cy) / s;
        b.seat = -1;
        let p, u, v;
        if (!isSide(b)) { p = dy > 0 ? 0 : 2; u = dy > 0 ? dx : -dx; v = Math.abs(dy); }
        else { p = dx > 0 ? 1 : 3; u = dx > 0 ? -dy : dy; v = Math.abs(dx); }
        if (v >= R0 && v <= R1 && u >= -UL && u <= UR && v > Math.abs(u) * 0.45) b.seat = p;
      }
    };
    assign();
    // 中心を河の内側の縁から推定し直す (河は中心から等距離に始まる)
    for (let it = 0; it < 2; it++) {
      const q = [0, 1, 2, 3].map((p) => cand.filter((b) => b.seat === p));
      const inner = [
        q[0].length >= 2 ? Math.min(...q[0].map((b) => b.y - b.h / 2)) : null,
        q[1].length >= 2 ? Math.min(...q[1].map((b) => b.x - b.w / 2)) : null,
        q[2].length >= 2 ? Math.max(...q[2].map((b) => b.y + b.h / 2)) : null,
        q[3].length >= 2 ? Math.max(...q[3].map((b) => b.x + b.w / 2)) : null,
      ];
      let nx = null, ny = null;
      if (inner[1] !== null && inner[3] !== null) nx = (inner[1] + inner[3]) / 2;
      if (inner[0] !== null && inner[2] !== null) ny = (inner[0] + inner[2]) / 2;
      // 片側しか無い軸は、もう一方の軸から分かった「中心〜河の距離」を使う
      const dH = nx !== null ? (inner[1] - inner[3]) / 2 : null, dV = ny !== null ? (inner[0] - inner[2]) / 2 : null;
      const d = dH !== null ? dH : dV;
      if (nx === null && d !== null) nx = inner[1] !== null ? inner[1] - d : inner[3] !== null ? inner[3] + d : null;
      if (ny === null && d !== null) ny = inner[0] !== null ? inner[0] - d : inner[2] !== null ? inner[2] + d : null;
      if (nx !== null) cx = nx;
      if (ny !== null) cy = ny;
      assign();
    }
    // リーチ宣言牌 (河の中で 90° 回っている牌) は隣の牌の河に合わせる
    for (const b of cand) {
      const side = isSide(b);
      let own = 0; const vote = {};
      for (const n of cand) {
        if (n === b || n.seat < 0) continue;
        if (isSide(n) === side) { if (Math.abs(n.x - b.x) < (n.w + b.w) * 0.75 && Math.abs(n.y - b.y) < (n.h + b.h) * 0.75 && n.seat === b.seat) own++; continue; }
        const inRow = side ? Math.abs(n.y - b.y) < n.h * 0.45 && Math.abs(n.x - b.x) < (n.w + b.w) / 2 + n.w * 0.35
          : Math.abs(n.x - b.x) < n.w * 0.45 && Math.abs(n.y - b.y) < (n.h + b.h) / 2 + n.h * 0.35;
        if (inRow) vote[n.seat] = (vote[n.seat] || 0) + 1;
      }
      const v = Object.keys(vote);
      if (v.length === 1 && !own) b.fix = +v[0];
    }
    for (const b of cand) if (b.fix !== undefined) { b.seat = b.fix; delete b.fix; }
    // 河は中心側から連続した塊。離れた塊 (副露やドラ表示など) は河から外す
    for (let p = 0; p < 4; p++) {
      const q = cand.filter((b) => b.seat === p);
      if (q.length < 2) continue;
      const comp = new Array(q.length).fill(-1);
      let nc = 0;
      for (let i = 0; i < q.length; i++) {
        if (comp[i] >= 0) continue;
        comp[i] = nc;
        const st = [i];
        while (st.length) {
          const a = q[st.pop()];
          for (let j = 0; j < q.length; j++) {
            if (comp[j] >= 0) continue;
            const n = q[j];
            if (Math.abs(a.x - n.x) < (a.w + n.w) / 2 + s * 0.45 && Math.abs(a.y - n.y) < (a.h + n.h) / 2 + s * 0.45) { comp[j] = nc; st.push(j); }
          }
        }
        nc++;
      }
      if (nc < 2) continue;
      let bi = 0, bd = Infinity;
      q.forEach((b, i) => { const d = Math.hypot(b.x - cx, b.y - cy); if (d < bd) { bd = d; bi = i; } });
      q.forEach((b, i) => { if (comp[i] !== comp[bi]) b.seat = -1; });
    }
    for (const b of rest) {
      const p = b.seat === undefined ? -1 : b.seat;
      delete b.seat;
      b.zone = p >= 0 ? 'river' + p : 'other';
      (p >= 0 ? rivers[p] : others).push(b);
    }
    // ドラ表示牌: 河でも手牌でもない正立の牌のうち、卓の左上にあるもの (同じ段で右に続く牌はカンドラ)
    const dc = others.filter((b) => b.w < b.h && b.conf >= 0.6 && b.x < cx - 4 * s && b.y < cy - 2.5 * s);
    let dora = [];
    if (dc.length) {
      dc.sort((a, b) => (a.x + a.y) - (b.x + b.y));
      const d0 = dc[0];
      dora = dc.filter((b) => Math.abs(b.y - d0.y) < d0.h * 0.35 && b.h > d0.h * 0.8 && b.h < d0.h * 1.25).sort((a, b) => a.x - b.x);
      let n = 1;
      while (n < dora.length && dora[n].x - dora[n - 1].x < d0.w * 1.4) n++;
      dora = dora.slice(0, Math.min(n, 5));
      for (const b of dora) b.zone = 'dora';
    }
    return { hand, rivers, others, dora, center: { x: cx, y: cy }, s };
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = V;
  else root.Vision = V;
})(typeof self !== 'undefined' ? self : this);
