"""牌位置検出 (CenterNet 風の小型 FCN) の学習。 usage: python3 train_det.py DATA_DIR OUT.pt MINUTES"""
import sys, os, glob, time, math, random
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F

def cbr(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

class Res(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.a, s.b = cbr(c, c), nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))
    def forward(s, x):
        return F.relu(x + s.b(s.a(x)))

class Det(nn.Module):
    """入力: RGB 0..1 (N,3,H,W)。出力: stride 4 の (N,5,H/4,W/4) = [heat(sigmoid済), dx, dy, log(w/16), log(h/16)]"""
    def __init__(s, c=(12, 24, 48, 96)):
        super().__init__()
        s.stem = nn.Sequential(cbr(3, c[0], 2), cbr(c[0], c[1], 2), Res(c[1]))
        s.d8 = nn.Sequential(cbr(c[1], c[2], 2), Res(c[2]), Res(c[2]))
        s.d16 = nn.Sequential(cbr(c[2], c[3], 2), Res(c[3]), Res(c[3]))
        s.l8 = nn.Conv2d(c[3], c[2], 1)
        s.f8 = cbr(c[2], c[2])
        s.l4 = nn.Conv2d(c[2], c[1], 1)
        s.f4 = cbr(c[1], c[1])
        s.head = nn.Sequential(cbr(c[1], c[1]), nn.Conv2d(c[1], 5, 1))
        s.head[1].bias.data[0] = -2.2
    def raw(s, x):
        x4 = s.stem(x)
        x8 = s.d8(x4)
        x16 = s.d16(x8)
        y8 = s.f8(x8 + F.interpolate(s.l8(x16), size=x8.shape[2:], mode='nearest'))
        y4 = s.f4(x4 + F.interpolate(s.l4(y8), size=x4.shape[2:], mode='nearest'))
        return s.head(y4)
    def forward(s, x):
        o = s.raw(x)
        return torch.cat([torch.sigmoid(o[:, :1]), o[:, 1:]], 1)

CROP, STR = 256, 4

def load(d, limit=None):
    """d: 'dir' または 'dirA:枚数,dirB' のようにカンマ区切りで複数指定"""
    fs = []
    for part in d.split(','):
        p, _, n = part.partition(':')
        fs += [f for f in sorted(glob.glob(p + '/*.jpg')) if os.path.exists(f[:-3] + 'npy')][:int(n) if n else limit]
    imgs = np.zeros((len(fs), 360, 640, 3), np.uint8)
    labs = []
    for i, f in enumerate(fs):
        imgs[i] = cv2.imread(f, cv2.IMREAD_REDUCED_COLOR_2)
        l = np.load(f[:-3] + 'npy').copy()
        l[:, :4] /= 2
        labs.append(l)
    return imgs, labs

def sample(imgs, labs, rnd):
    i = rnd.randrange(len(imgs))
    sc = rnd.uniform(0.7, 1.45)
    src = int(CROP / sc)
    H, W = imgs[i].shape[:2]
    x0, y0 = rnd.randint(0, W - src), rnd.randint(0, H - src) if src < H else 0
    if src >= H:
        src = H; sc = CROP / src; x0 = rnd.randint(0, W - src)
    im = imgs[i][y0:y0 + src, x0:x0 + src]
    im = cv2.resize(im, (CROP, CROP), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    l = labs[i].copy()
    l[:, 0] = (l[:, 0] - x0) * sc; l[:, 1] = (l[:, 1] - y0) * sc; l[:, 2:4] *= sc
    G = CROP // STR
    t = np.zeros((5, G, G), np.float32); m = np.zeros((G, G), np.float32)
    for cx, cy, w, h, c, o in l[:, :6]:
        gx, gy = cx / STR, cy / STR
        ix, iy = int(gx), int(gy)
        if w < 5 or h < 5:
            continue
        sg = min(3.0, max(0.6, min(w, h) / STR / 6))
        rad = int(3 * sg) + 1
        xa, xb, ya, yb = max(0, ix - rad), min(G, ix + rad + 1), max(0, iy - rad), min(G, iy + rad + 1)
        if xa >= xb or ya >= yb:
            continue
        yy, xx = np.mgrid[ya:yb, xa:xb]
        g = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sg * sg))
        t[0, ya:yb, xa:xb] = np.maximum(t[0, ya:yb, xa:xb], g)
        if 0 <= ix < G and 0 <= iy < G:
            t[0, iy, ix] = 1
            t[1:, iy, ix] = [gx - ix - 0.5, gy - iy - 0.5, math.log(w / 16), math.log(h / 16)]
            m[iy, ix] = 1
    return im, t, m

def batch(imgs, labs, rnd, n):
    xs, ts, ms = zip(*[sample(imgs, labs, rnd) for _ in range(n)])
    x = torch.from_numpy(np.stack(xs)[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float() / 255
    return x, torch.from_numpy(np.stack(ts)), torch.from_numpy(np.stack(ms))

def loss_fn(o, t, m):
    p = torch.sigmoid(o[:, 0]).clamp(1e-4, 1 - 1e-4)
    g = t[:, 0]
    pos = (g == 1).float()
    lh = -(pos * (1 - p) ** 2 * torch.log(p) + (1 - pos) * (1 - g) ** 4 * p ** 2 * torch.log(1 - p)).sum() / pos.sum().clamp(min=1)
    lr = (F.smooth_l1_loss(o[:, 1:], t[:, 1:], reduction='none', beta=0.2) * m[:, None]).sum() / m.sum().clamp(min=1)
    return lh + lr, lh, lr

def decode(o, thr=0.3):
    """o: (5,h,w) numpy(sigmoid済) → [(cx,cy,w,h,score)]"""
    h = o[0]
    mx = cv2.dilate(h, np.ones((3, 3), np.uint8))
    ys, xs = np.where((h >= mx) & (h > thr))
    return [((x + 0.5 + o[1, y, x]) * STR, (y + 0.5 + o[2, y, x]) * STR, 16 * math.exp(o[3, y, x]), 16 * math.exp(o[4, y, x]), h[y, x]) for y, x in zip(ys, xs)]

def evaluate(net, imgs, labs, thr=0.3):
    net.eval(); tp = fp = fn = 0
    with torch.no_grad():
        for im, l in zip(imgs, labs):
            x = torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
            d = decode(net(x)[0].numpy(), thr)
            used = set()
            for cx, cy, w, h, c, o in l[:, :6]:
                best = None
                for j, q in enumerate(d):
                    if j not in used and abs(q[0] - cx) < w * 0.3 and abs(q[1] - cy) < h * 0.3 and 0.7 < q[2] / w < 1.4 and 0.7 < q[3] / h < 1.4:
                        best = j; break
                if best is None: fn += 1
                else: used.add(best); tp += 1
            fp += len(d) - len(used)
    net.train()
    return tp / max(1, tp + fn), tp / max(1, tp + fp)

if __name__ == '__main__':
    data, out, minutes = sys.argv[1], sys.argv[2], float(sys.argv[3])
    torch.set_num_threads(int(__import__("os").environ.get("NT", "4")))
    imgs, labs = load(os.environ.get('TRAIN_DIRS', data + '/train'))
    vi, vl = load(os.environ.get('VAL_DIRS', data + '/val'), 60)
    print('loaded', len(imgs), flush=True)
    net = Det()
    if len(sys.argv) > 4:
        net.load_state_dict(torch.load(sys.argv[4]))
    print('params', sum(p.numel() for p in net.parameters()))
    net = net.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    rnd = random.Random(0)
    t0 = time.time(); it = 0; ema = None
    while True:
        el = (time.time() - t0) / 60
        if el > minutes: break
        lr = float(os.environ.get('LR', '2e-3')) * (0.5 * (1 + math.cos(math.pi * el / minutes))) + 2e-5
        if it < 200: lr *= (it + 1) / 200
        for g in opt.param_groups: g['lr'] = lr
        x, t, m = batch(imgs, labs, rnd, 16)
        l, lh, lr_ = loss_fn(net.raw(x.contiguous(memory_format=torch.channels_last)), t, m)
        opt.zero_grad(); l.backward(); opt.step()
        ema = l.item() if ema is None else 0.98 * ema + 0.02 * l.item()
        it += 1
        if it % 500 == 0:
            r, p = evaluate(net, vi, vl)
            print(f'it {it} {el:.1f}min loss {ema:.3f} recall {r:.3f} prec {p:.3f}', flush=True)
            torch.save(net.state_dict(), out)
    r, p = evaluate(net, vi, vl)
    print('final', r, p)
    torch.save(net.state_dict(), out)
