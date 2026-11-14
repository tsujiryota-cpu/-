"""牌種分類 (小型 CNN, 34種+非牌) の学習。 usage: python3 train_cls.py DATA_DIR OUT.pt MINUTES"""
import sys, os, glob, time, math, random
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from multiprocessing import Pool

CW, CH, NC = 32, 40, 35
PAD = 1.08

class Cls(nn.Module):
    """入力: RGB 0..1 (N,3,40,32)。出力: softmax 確率 (N,35)。class 34 = 牌ではない"""
    def __init__(s, c=(24, 48, 48, 96, 96, 128), fc=256):
        super().__init__()
        def cbr(i, o, st=1):
            return nn.Sequential(nn.Conv2d(i, o, 3, st, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        s.f = nn.Sequential(cbr(3, c[0]), cbr(c[0], c[1], 2), cbr(c[1], c[2]), cbr(c[2], c[3], 2), cbr(c[3], c[4]), cbr(c[4], c[5], 2), nn.Flatten(),
                            nn.Dropout(0.2), nn.Linear(c[5] * 5 * 4, fc), nn.ReLU(inplace=True), nn.Linear(fc, NC))
    def raw(s, x):
        return s.f(x)
    def forward(s, x):
        return torch.softmax(s.f(x), 1)

def crop(im, cx, cy, w, h, rot):
    """軸平行の箱を切り出し 32x40 に。rot=1/3: 横向きの箱を 90° 回して縦にする"""
    x0, y0, x1, y1 = cx - w * PAD / 2, cy - h * PAD / 2, cx + w * PAD / 2, cy + h * PAD / 2
    if rot == 0:
        src = np.float32([[x0, y0], [x1, y0], [x0, y1]])
    elif rot == 1:   # 画像の右辺が上になるよう回す
        src = np.float32([[x1, y0], [x1, y1], [x0, y0]])
    else:            # 左辺が上
        src = np.float32([[x0, y1], [x0, y0], [x1, y1]])
    M = cv2.getAffineTransform(src, np.float32([[0, 0], [CW, 0], [0, CH]]))
    return cv2.warpAffine(im, M, (CW, CH), flags=cv2.INTER_AREA if w * PAD > CW * 1.5 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def extract(f):
    rnd = random.Random(hash(f) & 0xffffff)
    im = cv2.imread(f)
    lab = np.load(f[:-3] + 'npy')[:, :6]
    H, W = im.shape[:2]
    X, Y = [], []
    idx = list(range(len(lab)))
    rnd.shuffle(idx)
    j = lambda s: rnd.uniform(-s, s)
    for i in idx[:44]:
        cx, cy, w, h, c, o = lab[i]
        if w < 9 or h < 9:
            continue
        side = w > h
        t = rnd.random()
        cx2, cy2, w2, h2 = cx + j(0.08) * w, cy + j(0.08) * h, w * (1 + j(0.1)), h * (1 + j(0.1))
        if t < 0.86:
            X.append(crop(im, cx2, cy2, w2, h2, rnd.choice([1, 2]) if side else 0)); Y.append(int(c))
        elif t < 0.93:  # 向き違い → 非牌
            if side:
                X.append(crop(im, cx2, cy2, w2, h2, 0))
            else:
                X.append(crop(im, cx2, cy2, w2, h2, rnd.choice([1, 2])))
            Y.append(34)
        else:           # 半分ずれ → 非牌
            dx, dy = rnd.choice([(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5), (0.5, 0.5), (-0.5, -0.5)])
            X.append(crop(im, cx + dx * w, cy + dy * h, w2, h2, rnd.choice([1, 2]) if side else 0)); Y.append(34)
    for _ in range(5):  # 背景
        w = rnd.uniform(14, 80); h = w * rnd.uniform(1.15, 1.45)
        if rnd.random() < 0.3: w, h = h, w
        cx, cy = rnd.uniform(0, W), rnd.uniform(0, H)
        if len(lab) and np.any((np.abs(lab[:, 0] - cx) < (lab[:, 2] + w) * 0.35) & (np.abs(lab[:, 1] - cy) < (lab[:, 3] + h) * 0.35)):
            continue
        X.append(crop(im, cx, cy, w, h, rnd.choice([1, 2]) if w > h else 0)); Y.append(34)
    return np.stack(X), np.array(Y, np.int64)

def build(d, limit=None):
    fs = []
    for part in d.split(','):
        fs += [f for f in sorted(glob.glob(part + '/*.jpg')) if os.path.exists(f[:-3] + 'npy')][:limit]
    with Pool(4) as p:
        r = p.map(extract, fs, chunksize=32)
    return np.concatenate([a for a, _ in r]), np.concatenate([b for _, b in r])

def to_t(x):
    return torch.from_numpy(x[:, :, :, ::-1].copy()).permute(0, 3, 1, 2).float() / 255

if __name__ == '__main__':
    data, out, minutes = sys.argv[1], sys.argv[2], float(sys.argv[3])
    torch.set_num_threads(int(__import__("os").environ.get("NT", "4")))
    X, Y = build(os.environ.get('TRAIN_DIRS', data + '/train'))
    VX, VY = build(os.environ.get('VAL_DIRS', data + '/val'))
    print('crops', X.shape, VX.shape, np.bincount(Y, minlength=NC), flush=True)
    np.savez_compressed(out + '.val.npz', x=VX[:2000], y=VY[:2000])
    net = Cls().to(memory_format=torch.channels_last)
    print('params', sum(p.numel() for p in net.parameters()))
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=5e-4)
    t0 = time.time(); it = 0; ema = None
    vx = to_t(VX)
    while True:
        el = (time.time() - t0) / 60
        if el > minutes: break
        lr = 2e-3 * (0.5 * (1 + math.cos(math.pi * el / minutes))) + 1e-5
        for g in opt.param_groups: g['lr'] = lr
        i = np.random.randint(0, len(X), 128)
        x = to_t(X[i])
        x = x * torch.empty(128, 1, 1, 1).uniform_(0.7, 1.2) + torch.empty(128, 3, 1, 1).uniform_(-0.08, 0.08)
        l = F.cross_entropy(net.raw(x.contiguous(memory_format=torch.channels_last)), torch.from_numpy(Y[i]), label_smoothing=0.05)
        opt.zero_grad(); l.backward(); opt.step()
        ema = l.item() if ema is None else 0.98 * ema + 0.02 * l.item()
        it += 1
        if it % 1000 == 0:
            net.eval()
            with torch.no_grad():
                p = torch.cat([net(vx[k:k + 1000]) for k in range(0, len(vx), 1000)]).argmax(1).numpy()
            net.train()
            tile = VY < 34
            print(f'it {it} {el:.1f}min loss {ema:.3f} acc {np.mean(p == VY):.4f} tile-acc {np.mean(p[tile] == VY[tile]):.4f} junk-acc {np.mean(p[~tile] == 34):.4f}', flush=True)
            torch.save(net.state_dict(), out)
    torch.save(net.state_dict(), out)
