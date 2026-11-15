"""合成データ生成: 麻雀ゲーム画面をカメラで撮影した風の画像と牌ラベルを作る。
usage: python3 gen.py OUT_DIR N [SEED0]
牌画像は FluffyStuff/riichi-mahjong-tiles (CC0) の SVG を使う (TILE_DIR で指定)。"""
import os, sys, io, random, math
import numpy as np, cv2

TILE_DIR = os.environ.get('TILE_DIR', '/home/user/FluffyStuff/riichi-mahjong-tiles/Regular/')
GW, GH = 150, 200
_Z = ['Ton', 'Nan', 'Shaa', 'Pei', 'Haku', 'Hatsu', 'Chun']
_S = ['Man', 'Pin', 'Sou']
_glyph = {}

ALT_DIR = os.environ.get('ALT_TILE_DIR', '/home/user/ext/mahjong-tiles-zddd312/tiles/')  # 別の絵柄 (任意)
NSETS = 1

REAL_DIR = os.environ.get('REAL_DIR', '')  # 実ゲームのスクリーンショット real*.png と ラベル real*.json (任意)

def _from_full(path, margin=(0.09, 0.07)):
    """牌全体の画像 (白地) から図柄だけを RGBA で取り出す。path は画像ファイルか RGB 配列"""
    from PIL import Image
    pil = Image.open(path) if isinstance(path, str) else Image.fromarray(path)
    im = np.array(pil.convert('RGBA').resize((GW, GH), Image.LANCZOS)).astype(np.float32)
    lum = im[:, :, :3].sum(2)
    face = im[:, :, :3][lum >= np.percentile(lum, 80)].mean(0)
    a = np.clip((np.abs(im[:, :, :3] - face).max(2) - 14) / 50, 0, 1) * (im[:, :, 3] / 255)
    m = np.zeros((GH, GW), np.float32)
    bx, by = int(GW * margin[0]), int(GH * margin[1])
    m[by:GH - by, bx:GW - bx] = 1
    im[:, :, 3] = a * m * 255
    return im.astype(np.uint8)

def _font_tile(c, r):
    """フォントで描いた萬子・字牌 (書体の多様性用)"""
    from PIL import Image, ImageDraw, ImageFont
    fonts = ['/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']
    fp = fonts[c % 2] if os.path.exists(fonts[c % 2]) else fonts[0]
    im = Image.new('RGBA', (GW, GH), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    red, grn, blk = (190, 30, 40, 255), (20, 120, 60, 255), (20, 20, 30, 255)
    def txt(ch, cy, size, col):
        f = ImageFont.truetype(fp, size)
        x0, y0, x1, y1 = d.textbbox((0, 0), ch, font=f)
        for dx in (0, 1, 2):  # 少し太らせる
            d.text((GW / 2 - (x0 + x1) / 2 + dx - 1, cy - (y0 + y1) / 2), ch, font=f, fill=col)
    if c < 9:
        txt('一二三四五六七八九'[c], GH * 0.27, 84, blk)
        txt('萬', GH * 0.70, 92, red)
    elif c == 31:
        d.rectangle((22, 26, GW - 22, GH - 26), outline=(30, 60, 160, 255), width=7)
    else:
        txt('東南西北白發中'[c - 27], GH * 0.5, 128, red if c == 33 else grn if c == 32 else blk)
    return np.array(im)

def load_glyphs():
    global NSETS
    import cairosvg
    from PIL import Image
    def rd(n):
        png = cairosvg.svg2png(url=TILE_DIR + n + '.svg', output_width=GW, output_height=GH)
        return np.array(Image.open(io.BytesIO(png)).convert('RGBA'))
    for c in range(34):
        n = _S[c // 9] + str(c % 9 + 1) if c < 27 else _Z[c - 27]
        _glyph[(0, c)] = rd(n)
    for s in range(3):
        _glyph[(0, 100 + s)] = rd(_S[s] + '5-Dora')
    if os.environ.get('MULTI') and os.path.isdir(ALT_DIR):
        zmap = {27: 'z1', 28: 'z2', 29: 'z3', 30: 'z4', 31: 'z7', 32: 'z6', 33: 'z5'}
        for c in range(34):
            _glyph[(1, c)] = _from_full(ALT_DIR + ('%d%s' % (c % 9 + 1, 'mps'[c // 9]) if c < 27 else zmap[c]) + '.png')
            _glyph[(2, c)] = _font_tile(c, None) if (c < 9 or c >= 27) else _glyph[(c % 2, c)]
        for k in (1, 2):
            for s in range(3):   # 赤五: 図柄を赤く塗る
                g = _glyph[(k, 4 + 9 * s)].copy()
                g[:, :, :3] = (0.25 * g[:, :, :3] + 0.75 * np.array([205, 35, 35])).astype(np.uint8)
                _glyph[(k, 100 + s)] = g
        NSETS = 3
        if REAL_DIR:
            import json, glob
            best = {}
            for jf in sorted(glob.glob(REAL_DIR + '/real*.json')):
                im = cv2.imread(jf[:-4] + 'png')[:, :, ::-1]
                for cx, cy, w, h, c, o, z in json.load(open(jf)):
                    crop = im[int(cy - h / 2) + 1:int(cy + h / 2) - 1, int(cx - w / 2) + 1:int(cx + w / 2) - 1]
                    crop = np.rot90(crop, {2: 3, 3: 2, 4: 1}.get(int(z), 0))
                    best.setdefault(int(c), []).append((w * h, np.ascontiguousarray(crop)))
            for c in range(34):
                if c in best:
                    v = sorted(best[c], key=lambda x: -x[0])
                    if c == 4 and len(v) > 1:      # 手牌の左端の五萬は赤五
                        _glyph[(3, 100)] = _from_full(v[0][1], (0.03, 0.03)); v = v[1:]
                    _glyph[(3, c)] = _from_full(v[0][1], (0.03, 0.03))
                else:
                    _glyph[(3, c)] = _glyph[(1, c)]
            for sidx in range(3):
                _glyph.setdefault((3, 100 + sidx), _glyph[(1, 100 + sidx)])
            NSETS = 4

def rot_pts(pts, w, k):
    """np.rot90(img,k) と同じ回転を点に適用。w: 回転前の画像幅(正方形前提なら一辺)"""
    pts = pts.copy()
    for _ in range(k % 4):
        x, y = pts[:, 0].copy(), pts[:, 1].copy()
        pts[:, 0], pts[:, 1] = y, w - x
    return pts

class Style:
    def __init__(s, r):
        v = r.randint(215, 255)
        s.face = np.array([v - r.randint(0, 28), v - r.randint(0, 12), v], np.float32)  # BGR, 白〜クリーム
        s.border = r.uniform(0.25, 0.8)
        s.radius = r.uniform(0.04, 0.16)
        s.thick = r.choice([0, 0, r.uniform(0.05, 0.22)])
        s.side = np.array(r.choice([(60, 170, 235), (200, 140, 50), (90, 160, 70), (200, 200, 200), (60, 120, 220), (120, 120, 120)]), np.float32)
        s.side = np.clip(s.side + np.array([r.randint(-30, 30) for _ in range(3)]), 0, 255)
        s.gscale = r.uniform(0.78, 0.98)
        s.ratio = r.uniform(1.22, 1.42)
        s.back = np.array([r.randint(20, 255) for _ in range(3)], np.float32)
        s.gset = r.choice([0, 0, 1, 1, 2, 3, 3, 3] if NSETS > 3 else [0, 0, 0, 1, 1, 1, 2]) if NSETS > 1 else 0
        s.ms = NSETS > 3 and r.random() < 0.6   # 雀魂風: 厚みのある牌・牌山・中央の点数表示・近い河
        if s.ms:
            s.thick = r.uniform(0.12, 0.3)
            s.back = np.clip(np.array(r.choice([(40, 150, 245), (40, 150, 245), (200, 120, 40), (70, 160, 60), (150, 150, 150)]), np.float32) + np.array([r.randint(-25, 25) for _ in range(3)]), 0, 255)
            s.gray = np.float32([r.randint(170, 225)] * 3)
        s.cache = {}

def make_tile(code, w, st, r, back=False):
    """正立の牌スプライト RGBA(float32)。戻り値 (img, face_h)"""
    key = (code, w, back)
    if key in st.cache:
        return st.cache[key]
    h = max(4, int(round(w * st.ratio)))
    th = int(round(h * st.thick))
    img = np.zeros((h + th, w, 4), np.float32)
    rad = max(1, int(st.radius * w))
    def rrect(y0, y1, col):
        m = np.zeros((h + th, w), np.uint8)
        cv2.rectangle(m, (rad, y0), (w - 1 - rad, y1 - 1), 255, -1)
        cv2.rectangle(m, (0, y0 + rad), (w - 1, y1 - 1 - rad), 255, -1)
        for cx in (rad, w - 1 - rad):
            for cy in (y0 + rad, y1 - 1 - rad):
                cv2.circle(m, (cx, cy), rad, 255, -1)
        k = m > 0
        img[k, :3] = col
        img[k, 3] = 1
        return k
    if th and getattr(st, 'ms', False):
        if back:
            rrect(th, h + th, np.float32([235, 235, 235]))
        else:
            rrect(th, h + th, st.back)
            rrect(th // 2, h + th // 2, st.gray)
    elif th:
        rrect(th, h + th, st.side)
    k = rrect(0, h, st.back if back else st.face)
    if not back:
        g = _glyph[(st.gset, code)]
        gw, gh = max(2, int(w * st.gscale)), max(2, int(h * st.gscale))
        g = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA).astype(np.float32)
        ox, oy = (w - gw) // 2, (h - gh) // 2
        a = g[:, :, 3:4] / 255
        img[oy:oy + gh, ox:ox + gw, :3] = img[oy:oy + gh, ox:ox + gw, :3] * (1 - a) + g[:, :, 2::-1] * a
    # 輪郭線
    e = cv2.morphologyEx(k.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    img[e & k, :3] *= (1 - st.border * 0.5)
    st.cache[key] = (img, h)
    return img, h

def paste(dst, alpha, spr, x, y, dim=1.0):
    """dst: float32 BGR, alpha: None か float32 マスク"""
    h, w = spr.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4]
    dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - a) + s[:, :, :3] * dim * a
    if alpha is not None:
        alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], a[:, :, 0])

def put_tile(dst, alpha, labels, code, x, y, w, st, r, k=0, dim=1.0, back=False, zone=5):
    """(x,y) を左上として牌を置く。k: 反時計回り 90°×k。返り値: 置いた矩形の (w,h)"""
    c = code if code < 100 else 4 + 9 * (code - 100)
    spr, fh = make_tile(code, w, st, r, back)
    quad = np.float32([[0, 0], [w, 0], [w, fh], [0, fh]])
    if k % 4:
        H0, W0 = spr.shape[:2]
        side = max(H0, W0)
        quad = rot_pts(quad, W0, 1) if k % 4 == 1 else quad
        if k % 4 == 2:
            quad = np.float32([[W0 - q[0], H0 - q[1]] for q in quad])
        if k % 4 == 3:
            quad = np.float32([[H0 - q[1], q[0]] for q in quad])
        spr = np.rot90(spr, k % 4)
    paste(dst, alpha, spr, x, y, dim)
    if not back:
        labels.append((c, quad + np.float32([x, y]), zone))
    return spr.shape[1], spr.shape[0]

def wall(r):
    w = [c for c in range(34) for _ in range(4)]
    r.shuffle(w)
    return w

def aka(c, r):
    return 100 + c // 9 if (c < 27 and c % 9 == 4 and r.random() < 0.2) else c

def background(W, H, r):
    pal = [(60, 110, 30), (90, 70, 20), (40, 60, 90), (30, 30, 30), (110, 90, 60), (140, 150, 150), (70, 40, 90)]
    c1 = np.array(r.choice(pal), np.float32) + r.randint(-20, 40)
    c2 = np.array(r.choice(pal), np.float32) + r.randint(-20, 40)
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None] if r.random() < 0.5 else np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    img = np.clip(c1 * (1 - t) + c2 * t, 0, 255) * np.ones((H, W, 1), np.float32)
    if r.random() < 0.5:
        n = cv2.resize(np.random.rand(H // 8, W // 8).astype(np.float32), (W, H)) - 0.5
        img += n[:, :, None] * r.uniform(5, 40)
    return img

def clutter(dst, r, n):
    if not n:
        return
    img = np.clip(dst, 0, 255).astype(np.uint8)
    _clutter(img, r, n)
    dst[:] = img

def _clutter(img, r, n):
    H, W = img.shape[:2]
    for _ in range(n):
        col = [r.randint(0, 255) for _ in range(3)] if r.random() < 0.6 else [r.randint(215, 255)] * 3
        x, y = r.randint(0, W), r.randint(0, H)
        t = r.random()
        if t < 0.35:
            cv2.rectangle(img, (x, y), (x + r.randint(8, 200), y + r.randint(8, 90)), col, -1 if r.random() < 0.7 else 2)
        elif t < 0.5:
            cv2.circle(img, (x, y), r.randint(5, 60), col, -1)
        elif t < 0.85:
            s = ''.join(r.choice('0123456789ABCDEFGHKLMNPRSTUXYZ+-:.') for _ in range(r.randint(1, 8)))
            cv2.putText(img, s, (x, y), r.choice([0, 2, 3, 4, 6]), r.uniform(0.4, 2.2), col, r.randint(1, 4), cv2.LINE_AA)
        else:
            cv2.line(img, (x, y), (r.randint(0, W), r.randint(0, H)), col, r.randint(1, 4))

def scene(seed):
    r = random.Random(seed)
    np.random.seed(seed % (2 ** 31))
    st = Style(r)
    W, H = r.choice([1280, 1280, 1400, 1560]), 720
    img = background(W, H, r)
    clutter(img, r, r.randint(0, 14))
    labels = []
    wl = wall(r)
    # ---- 卓レイヤ (河・副露・相手の手牌)
    S = 720
    tl = np.zeros((S, S, 3), np.float32)
    ta = np.zeros((S, S), np.float32)
    tlab = []
    rw = r.randint(26, 42) if st.ms else r.randint(22, 40)
    rh = int(round(rw * st.ratio))
    d0 = (r.uniform(2.3, 3.0) if st.ms else r.uniform(3.0, 4.0)) * rw
    if st.ms and r.random() < 0.85:   # 中央の点数表示
        q = int(d0 - 0.3 * rw)
        box = np.zeros((2 * q, 2 * q, 3), np.uint8)
        box[:] = [r.randint(40, 110)] * 3
        _clutter(box, r, r.randint(3, 8))
        cv2.rectangle(box, (0, 0), (2 * q - 1, 2 * q - 1), [r.randint(20, 80)] * 3, r.randint(2, 6))
        tl[S // 2 - q:S // 2 + q, S // 2 - q:S // 2 + q] = box
        ta[S // 2 - q:S // 2 + q, S // 2 - q:S // 2 + q] = 1
    full = r.random() < 0.85
    for p in range(4):
        lay = np.zeros((S, S, 3), np.float32)
        la = np.zeros((S, S), np.float32)
        ll = []
        n = r.randint(0, 22) if full else r.randint(0, 3)
        ri = r.randint(0, max(0, n - 1)) if r.random() < 0.35 else -1
        x0 = S / 2 - 3 * rw + r.uniform(-0.3, 0.3) * rw
        x, y = x0, S / 2 + d0
        pitch = rh + int(rh * st.thick * r.uniform(0, 0.6)) + r.randint(0, 2)
        for i in range(n):
            if i and i % 6 == 0 and i < 18:
                x, y = x0, y + pitch
            c = aka(wl.pop(), r)
            dim = r.choice([1, 1, 1, 1, r.uniform(0.5, 0.85)])
            if i == ri:
                w_, h_ = put_tile(lay, la, ll, c, int(x), int(y + rh - rw), rw, st, r, 1, dim, zone=1 + p)
            else:
                w_, h_ = put_tile(lay, la, ll, c, int(x), int(y), rw, st, r, 0, dim, zone=1 + p)
            x += w_ + r.choice([0, 0, 1])
        # 副露 (右下)
        mw = r.randint(22, 40)
        mx = S - r.randint(5, 40)
        my = S - int(mw * st.ratio) - r.randint(4, 60)
        for _ in range(r.choice([0, 0, 1, 1, 2, 3, 4]) if full else 0):
            base = wl.pop()
            kind = r.random()
            cs = [base] * (4 if kind < 0.15 else 3)
            if kind > 0.6 and base < 27 and base % 9 <= 6:
                cs = [base, base + 1, base + 2]
            side = r.randint(0, len(cs) - 1)
            for j, c in enumerate(cs):
                k = 1 if j == side else 0
                tw = int(mw * st.ratio) if k else mw
                mx -= tw
                mh = int(mw * st.ratio)
                put_tile(lay, la, ll, aka(c, r), mx, my + (mh - mw if k else 0), mw, st, r, k, back=(len(cs) == 4 and kind < 0.05 and j in (0, 3)))
            mx -= r.randint(2, 10)
        # 相手の手牌 (裏)
        if p and r.random() < (0.85 if st.ms else 0.6):
            bw = r.randint(18, 34)
            bx = S / 2 - 7 * bw
            for j in range(r.randint(6, 17) if st.ms else r.choice([13, 10, 7])):
                put_tile(lay, la, ll, 0, int(bx + j * bw), S - int(bw * st.ratio) - 2 if r.random() < 2 else 0, bw, st, r, 0, back=True)
        if p:
            lay, la = np.rot90(lay, p), np.rot90(la, p)
            ll = [(c, rot_pts(q, S, p), z) for c, q, z in ll]
        a3 = la[:, :, None]
        tl = tl * (1 - a3) + lay * a3
        ta = np.maximum(ta, la)
        tlab += ll
    # 卓の傾き
    top = r.uniform(0.72, 1.0) if r.random() < 0.6 else 1.0
    vs = r.uniform(0.7, 1.0) if top < 1 else r.uniform(0.9, 1.0)
    if st.ms:
        top, vs = r.uniform(0.62, 0.88), r.uniform(0.58, 0.82)
    sc = r.uniform(0.85, 1.0) * (H * 0.95 / S)
    cx, cy = W / 2 + r.uniform(-60, 60), H * r.uniform(0.40, 0.48)
    hw, hh = S * sc / 2, S * sc * vs / 2
    dst = np.float32([[cx - hw * top, cy - hh], [cx + hw * top, cy - hh], [cx + hw, cy + hh], [cx - hw, cy + hh]])
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [S, 0], [S, S], [0, S]]), dst)
    wt = cv2.warpPerspective(tl, M, (W, H), flags=cv2.INTER_AREA if sc < 0.8 else cv2.INTER_LINEAR)
    wa = cv2.warpPerspective(ta, M, (W, H))[:, :, None]
    img = img * (1 - wa) + wt * wa
    for c, q, z in tlab:
        labels.append((c, cv2.perspectiveTransform(q[None], M)[0], z))
    # ---- ドラ表示
    if r.random() < 0.8:
        dw = r.randint(24, 44)
        dx, dy = r.randint(5, W // 3), r.randint(5, H // 4)
        nd = r.randint(1, 4)
        for j in range(5):
            put_tile(img, None, labels, aka(wl.pop(), r), dx + j * (dw + 1), dy, dw, st, r, 0, back=j >= nd)
    # ---- 自分の手牌
    n = r.choice([13, 14, 14, 14, 10, 11, 11, 7, 8, 4, 5, 1, 2])
    hw_ = r.randint(50, min(96, (W - 60) // 15))
    hand = sorted(wl.pop() for _ in range(n))
    if n % 3 == 2:
        j = r.randrange(n)
        hand.append(hand.pop(j))
    hx = r.randint(10, max(11, W - 15 * hw_ - 10))
    hy = H - int(hw_ * st.ratio * (1 + st.thick)) - r.randint(0, 40)
    sth = st
    if st.ms:      # 手牌は正面向きで厚みなし、下端が画面外に切れることがある
        import copy
        sth = copy.copy(st); sth.thick = 0; sth.cache = {}
        hy = H - int(hw_ * st.ratio * r.uniform(0.78, 1.05))
    sel = r.randrange(n) if r.random() < 0.3 else -1
    gap = r.choice([0, 1, 2])
    for i, c in enumerate(hand):
        if i == n - 1 and n % 3 == 2:
            hx += int(hw_ * r.uniform(0.15, 0.5))
        put_tile(img, None, labels, aka(c, r), hx, hy - (int(hw_ * 0.25) if i == sel else 0), hw_, sth, r, 0, r.choice([1, 1, 1, 1, 1, 0.6]), zone=0)
        hx += hw_ + gap
    # ばらまき牌 (多様性用)
    for _ in range(r.choice([0, 0, 0, 2, 5])):
        put_tile(img, None, labels, aka(r.randrange(34), r), r.randint(0, W - 60), r.randint(0, H - 80), r.randint(22, 70), st, r, r.randint(0, 3))
    clutter(img, r, r.randint(0, 3))
    return np.clip(img, 0, 255).astype(np.uint8), labels

def camera(img, labels, seed, OW=1280, OH=720):
    r = random.Random(seed * 7 + 1)
    H, W = img.shape[:2]
    fill = r.uniform(0.7, 1.04)
    s = min(OW / W, OH / H) * fill
    w2, h2 = W * s / 2, H * s / 2
    cx, cy = OW / 2 + r.uniform(-1, 1) * max(0, OW / 2 - w2), OH / 2 + r.uniform(-1, 1) * max(0, OH / 2 - h2)
    j = lambda: r.uniform(-0.035, 0.035) * OH
    dst = np.float32([[cx - w2 + j(), cy - h2 + j()], [cx + w2 + j(), cy - h2 + j()], [cx + w2 + j(), cy + h2 + j()], [cx - w2 + j(), cy + h2 + j()]])
    ang = r.uniform(-4, 4) if r.random() < 0.7 else 0
    R = cv2.getRotationMatrix2D((OW / 2, OH / 2), ang, 1)
    dst = cv2.transform(dst[None], R)[0]
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), dst)
    bg = background(OW, OH, r) * r.uniform(0.2, 0.8)
    clutter(bg, r, r.randint(0, 6))
    out = cv2.warpPerspective(img.astype(np.float32), M, (OW, OH), dst=bg, flags=cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)
    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    if r.random() < 0.5:  # モアレ
        f = r.uniform(0.3, 2.5)
        th = r.uniform(0, math.pi)
        out *= (1 + r.uniform(0.02, 0.12) * np.sin((xx * math.cos(th) + yy * math.sin(th)) * f))[:, :, None]
    if r.random() < 0.4:  # 映り込み
        gx, gy, gs = r.uniform(0, OW), r.uniform(0, OH), r.uniform(80, 400)
        out += (r.uniform(30, 120) * np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * gs * gs)))[:, :, None]
    out *= np.array([r.uniform(0.85, 1.1) for _ in range(3)], np.float32) * r.uniform(0.6, 1.15)
    out = 255 * np.clip(out / 255, 0, 1) ** r.uniform(0.7, 1.4)
    sg = r.choice([0, r.uniform(0.3, 1.2), r.uniform(0.5, 2.2)])
    if sg:
        out = cv2.GaussianBlur(out, (0, 0), sg)
    if r.random() < 0.15:
        k = r.randint(3, 7)
        ker = np.zeros((k, k), np.float32)
        ker[k // 2] = 1 / k
        if r.random() < 0.5:
            ker = ker.T
        out = cv2.filter2D(out, -1, ker)
    out += np.random.normal(0, r.uniform(0, 7), out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    _, enc = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, r.randint(45, 95)])
    lab = []
    for c, q, z in labels:
        q = cv2.perspectiveTransform(q[None].astype(np.float32), M)[0]
        ctr = q.mean(0)
        x0, y0 = q.min(0)
        x1, y1 = q.max(0)
        if not (0 <= ctr[0] < OW and 0 <= ctr[1] < OH):
            continue
        up = (q[0] + q[1]) / 2 - ctr
        o = (0 if up[1] < 0 else 2) if abs(up[1]) > abs(up[0]) else (1 if up[0] < 0 else 3)
        lab.append([ctr[0], ctr[1], x1 - x0, y1 - y0, c, o, z])
    return enc, np.float32(lab).reshape(-1, 7)

def real_frames(out, n, s0):
    """実スクリーンショット (REAL_DIR/real*.png + .json) をカメラ撮影風に加工して学習用フレームにする"""
    import json, glob
    os.makedirs(out, exist_ok=True)
    src = []
    for jf in sorted(glob.glob(REAL_DIR + '/real*.json')):
        lab = []
        for cx, cy, w, h, c, o, z in json.load(open(jf)):
            q = [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]
            k = {2: 3, 3: 2, 4: 1}.get(int(z), 0)   # 牌の上辺が q[0]-q[1] になるよう並べ替え (下家の河=左向き, 対面=下向き, 上家=右向き)
            lab.append((int(c), np.float32(q[k:] + q[:k]), int(z)))
        src.append((cv2.imread(jf[:-4] + 'png'), lab))
    for s in range(s0, s0 + n):
        img, lab = src[s % len(src)]
        enc, l = camera(img, lab, s)
        enc.tofile(f'{out}/{s:06d}.jpg')
        np.save(f'{out}/{s:06d}.npy', l)

def main():
    if sys.argv[1] == '--real':
        return real_frames(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    out, n = sys.argv[1], int(sys.argv[2])
    s0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(out, exist_ok=True)
    load_glyphs()
    for s in range(s0, s0 + n):
        img, labels = scene(s)
        enc, lab = camera(img, labels, s)
        enc.tofile(f'{out}/{s:06d}.jpg')
        np.save(f'{out}/{s:06d}.npy', lab)

if __name__ == '__main__':
    main()
