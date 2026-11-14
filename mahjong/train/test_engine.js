const MJ = require('../engine.js');
const assert = require('assert');
const sh = (s, m = 0) => MJ.shanten(MJ.parse(s), m);
assert.strictEqual(sh('123m456p789s11222z'), -1);
assert.strictEqual(sh('123m456p789s1122z'), 0);
assert.strictEqual(sh('1122334455667z'), 0);          // 七対聴牌
assert.strictEqual(sh('19m19p19s1234567z'), 0);       // 国士13面
assert.strictEqual(sh('19m19p19s12345677z'), -1);
assert.strictEqual(sh('1112345678999m'), 0);
assert.strictEqual(sh('11123456789999m'), -1);
assert.strictEqual(sh('123456789m1234p'), 0);
assert.strictEqual(sh('123456789m12p11z'), 0);
assert.strictEqual(sh('55m234p', 3), -1);
assert.strictEqual(sh('5m234p', 3), 0);
assert.strictEqual(sh('1m', 4), 0);
assert.strictEqual(sh('11m', 4), -1);
// 既知の値
assert.strictEqual(sh('147m258p369s1234z'), 6);
assert.strictEqual(sh('3466m23467p34s11z'), 2);
assert.strictEqual(sh('1199m1199p1199s1z'), 0);
// ランダムな和了形は必ず -1、1枚抜くと 0
function rndAgari(r) {
  for (;;) {
    const c = new Int8Array(34);
    const p = (r() * 34) | 0; c[p] += 2;
    for (let i = 0; i < 4; i++) {
      if (r() < 0.4) { c[(r() * 34) | 0] += 3; } else { const s = (r() * 3) | 0, n = (r() * 7) | 0; c[s * 9 + n]++; c[s * 9 + n + 1]++; c[s * 9 + n + 2]++; }
    }
    if (c.every((x) => x <= 4)) return c;
  }
}
let seed = 1; const r = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
for (let i = 0; i < 3000; i++) {
  const c = rndAgari(r);
  assert.strictEqual(MJ.shanten(c, 0), -1);
  const t = [...c.keys()].filter((k) => c[k])[0]; c[t]--;
  assert.strictEqual(MJ.shanten(c, 0), 0);
  const u = MJ.ukeire(c, 0, null);
  assert(u.tiles.some((x) => x.t === t));
}
// 打牌候補
let a = MJ.analyzeDiscard(MJ.parse('123m456p789s1123z5m'), {});
console.log('discard:', a.shanten, a.cands.slice(0, 4).map((x) => MJ.NAMES[x.t] + ':' + x.shanten + '/' + x.ukeire).join(' '));
a = MJ.analyzeDiscard(MJ.parse('2346m3378p4567s19m'.replace('19m', '') + '19m'), {});
console.log('discard:', a.shanten, a.cands.slice(0, 5).map((x) => MJ.NAMES[x.t] + ':' + x.shanten + '/' + x.ukeire).join(' '));
// 危険度
const river = new Int8Array(34); river[MJ.parse('4m').indexOf(1)] = 1; river[27] = 1;
const seen = new Int8Array(34); seen[3] = 1; seen[27] = 1;
const hand = MJ.parse('1m7m5m5p9s1z2z');
for (const t of [0, 6, 4, 13, 26, 27, 28]) console.log(MJ.NAMES[t], MJ.dangerOf(t, [{ river }], seen, hand), MJ.dangerLabel(MJ.dangerOf(t, [{ river }], seen, hand)));
a = MJ.analyzeDiscard(MJ.parse('2389m3378p4567s19m'.slice(0)), { threats: [{ river }], seen });
console.log(a.mode, MJ.NAMES[a.pick.t], a.note);
// 鳴き
let k = MJ.analyzeCalls(MJ.parse('23m4688p55z345s19m'), { winds: { round: 0, seat: 1 } });
console.log('calls', k.shanten, k.ukeire, k.calls.map((x) => x.kind + MJ.NAMES[x.t] + '→' + x.shanten + ' 切' + MJ.NAMES[x.discard] + ' ' + x.hints.join('/') + (x.good ? ' ◎' : ' △')).join(' | '));
// 速度
const t0 = Date.now();
for (let i = 0; i < 200; i++) { const c = rndAgari(r); c[(r() * 34) | 0] < 4 && MJ.analyzeDiscard(c, {}); }
console.log('200 analyzeDiscard ms', Date.now() - t0);
console.log('ok');
