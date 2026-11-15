// 麻雀の判定ロジック (向聴数 / 受け入れ / 打牌候補 / 鳴き判断 / 危険度)
// 牌は 0..33 の番号: 0-8 萬子, 9-17 筒子, 18-26 索子, 27-33 字牌(東南西北白發中)
(function (root) {
  'use strict';

  const NAMES = [];
  for (const s of 'mps') for (let n = 1; n <= 9; n++) NAMES.push(n + s);
  for (let n = 1; n <= 7; n++) NAMES.push(n + 'z');
  const KANJI = ['東', '南', '西', '北', '白', '發', '中'];
  const SUIT_JA = ['萬', '筒', '索'];
  const label = (t) => (t < 27 ? (t % 9 + 1) + SUIT_JA[(t / 9) | 0] : KANJI[t - 27]);

  // ドラ表示牌 → ドラ
  function doraOf(ind) {
    if (ind < 27) return ind - (ind % 9) + ((ind % 9) + 1) % 9;
    if (ind < 31) return 27 + ((ind - 27 + 1) % 4);
    return 31 + ((ind - 31 + 1) % 3);
  }
  const countDora = (c, dora) => { let n = 0; if (dora) for (let t = 0; t < 34; t++) n += c[t] * dora[t]; return n; };

  // ---------- 向聴数 ----------
  // 1色(9種)について [雀頭あり/なし][面子数 m] → 最大の搭子数 を返す (メモ化)
  const suitMemo = new Map();
  const NEG = -9;
  function suitTable(c, off) {
    let key = 0;
    for (let i = 0; i < 9; i++) key = key * 5 + c[off + i];
    let r = suitMemo.get(key);
    if (r) return r;
    r = [new Int8Array(5).fill(NEG), new Int8Array(5).fill(NEG)];
    const a = Array.from({ length: 9 }, (_, i) => c[off + i]);
    (function rec(i, m, t, head) {
      while (i < 9 && a[i] === 0) i++;
      if (i >= 9) {
        if (m <= 4 && t > r[head][m]) r[head][m] = t;
        return;
      }
      if (a[i] >= 3) { a[i] -= 3; rec(i, m + 1, t, head); a[i] += 3; }
      if (i < 7 && a[i + 1] && a[i + 2]) {
        a[i]--; a[i + 1]--; a[i + 2]--; rec(i, m + 1, t, head); a[i]++; a[i + 1]++; a[i + 2]++;
      }
      if (a[i] >= 2) {
        a[i] -= 2;
        rec(i, m, t + 1, head);
        if (!head) rec(i, m, t, 1);
        a[i] += 2;
      }
      if (i < 8 && a[i + 1]) { a[i]--; a[i + 1]--; rec(i, m, t + 1, head); a[i]++; a[i + 1]++; }
      if (i < 7 && a[i + 2]) { a[i]--; a[i + 2]--; rec(i, m, t + 1, head); a[i]++; a[i + 2]++; }
      const k = a[i]; a[i] = 0; rec(i + 1, m, t, head); a[i] = k; // 残りは孤立牌扱い
    })(0, 0, 0, 0);
    suitMemo.set(key, r);
    return r;
  }
  function honorTable(c) {
    let m = 0, p = 0;
    for (let i = 27; i < 34; i++) { if (c[i] >= 3) m++; else if (c[i] === 2) p++; }
    const r = [new Int8Array(5).fill(NEG), new Int8Array(5).fill(NEG)];
    if (m <= 4) { r[0][m] = p; if (p) r[1][m] = p - 1; }
    return r;
  }
  // 一般形の向聴数。melds: 副露(暗槓含む)の面子数
  function shantenNormal(c, melds) {
    let cur = [new Int8Array(5).fill(NEG), new Int8Array(5).fill(NEG)];
    cur[0][0] = 0;
    const tabs = [suitTable(c, 0), suitTable(c, 9), suitTable(c, 18), honorTable(c)];
    for (const tb of tabs) {
      const nx = [new Int8Array(5).fill(NEG), new Int8Array(5).fill(NEG)];
      for (let h1 = 0; h1 < 2; h1++) for (let m1 = 0; m1 < 5; m1++) {
        const t1 = cur[h1][m1];
        if (t1 < 0) continue;
        for (let h2 = 0; h2 + h1 < 2; h2++) for (let m2 = 0; m1 + m2 < 5; m2++) {
          const t2 = tb[h2][m2];
          if (t2 < 0) continue;
          if (t1 + t2 > nx[h1 + h2][m1 + m2]) nx[h1 + h2][m1 + m2] = t1 + t2;
        }
      }
      cur = nx;
    }
    let best = 8;
    for (let h = 0; h < 2; h++) for (let m = 0; m < 5; m++) {
      let t = cur[h][m];
      if (t < 0) continue;
      const M = m + melds;
      if (M > 4) continue;
      if (M + t > 4) t = 4 - M;
      const s = 8 - 2 * M - t - h;
      if (s < best) best = s;
    }
    return best;
  }
  function shantenChiitoi(c) {
    let pairs = 0, kinds = 0;
    for (let i = 0; i < 34; i++) { if (c[i] >= 2) pairs++; if (c[i]) kinds++; }
    return 6 - pairs + Math.max(0, 7 - kinds);
  }
  const YAOCHU = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33];
  function shantenKokushi(c) {
    let kinds = 0, pair = 0;
    for (const i of YAOCHU) { if (c[i]) kinds++; if (c[i] >= 2) pair = 1; }
    return 13 - kinds - pair;
  }
  function shanten(c, melds) {
    let s = shantenNormal(c, melds || 0);
    if (!melds) s = Math.min(s, shantenChiitoi(c), shantenKokushi(c));
    return s;
  }

  // ---------- 受け入れ ----------
  // c: 手牌(3n+1枚), seen: 自分から見えている枚数(手牌以外)。戻り値 {tiles:[{t,n}], total}
  function ukeire(c, melds, seen, s0) {
    if (s0 === undefined) s0 = shanten(c, melds);
    const tiles = [];
    let total = 0;
    for (let t = 0; t < 34; t++) {
      if (c[t] >= 4) continue;
      // 孤立した牌で隣接もなければ一般形では進まない → 七対/国士の可能性があるので全て試す(メモ化で十分速い)
      c[t]++;
      const s = shanten(c, melds);
      c[t]--;
      if (s < s0) {
        const n = Math.max(0, 4 - c[t] - (seen ? seen[t] : 0));
        tiles.push({ t, n });
        total += n;
      }
    }
    return { tiles, total };
  }

  // ---------- 危険度 ----------
  // threat: {river: Int8Array(34) その相手の捨て牌} / seenAll: 自分の手牌以外で見えている枚数 / hand: 自分の手牌
  // 戻り値 0(安全)〜100
  function dangerOne(t, threat, seenAll, hand) {
    if (threat.river[t]) return 0; // 現物
    const vis = (x) => (seenAll[x] || 0) + (hand[x] || 0);
    if (t >= 27) {
      const left = 4 - vis(t) + (hand[t] || 0) - (hand[t] || 0); // 自分以外に残っている枚数 = 4 - 見えている - 自分
      const unseen = 4 - vis(t);
      // 自分が k 枚持っていて残り unseen 枚: 相手が待てるのは 単騎(1枚) / シャンポン(2枚)
      if (unseen <= 0) return 1;
      if (unseen === 1) return 4;
      if (unseen === 2) return 10;
      return t >= 31 ? 20 : 16; // 生牌 (三元牌はやや高め)
    }
    const n = t % 9, base = t - n;
    const gen = (k) => k >= 0 && k <= 8 && threat.river[base + k] > 0;
    const wall = (k) => vis(base + k) >= 4;
    const thin = (k) => vis(base + k) === 3;
    // 両面待ち: (n+1,n+2) で n と n+3 待ち / (n-2,n-1) で n-3 と n 待ち
    let live = 0, weak = 0;
    if (n + 2 <= 8 && !(n + 1 === 8)) { // 上側の両面 (n+1,n+2)。n+2<=8 かつ 端(8,9)でない
      if (n + 2 <= 7 || n + 3 <= 8) {
        if (n + 3 <= 8) {
          if (!gen(n + 3) && !wall(n + 1) && !wall(n + 2)) { live++; if (thin(n + 1) || thin(n + 2)) weak++; }
        }
      }
    }
    if (n - 3 >= 0) {
      if (!gen(n - 3) && !wall(n - 1) && !wall(n - 2)) { live++; if (thin(n - 1) || thin(n - 2)) weak++; }
    }
    const d = Math.min(n, 8 - n); // 0:端(1,9) 1:(2,8) 2:(3,7) 3+:(4,5,6)
    let v;
    if (live === 0) v = [3, 5, 6, 5, 5][d];           // 筋・壁 (カンチャン/ペンチャン/単騎/シャンポンのみ)
    else if (live === 1) v = [12, 15, 18, 20, 20][d]; // 片筋 or 端寄りの無筋
    else v = 30;                                       // 無筋 456
    if (weak && live) v *= weak >= live ? 0.6 : 0.8;   // ワンチャンス
    // シャンポン/単騎が物理的に薄い場合はさらに下げる
    const unseen = 4 - vis(t);
    if (live === 0 && unseen <= 1) v *= 0.5;
    return Math.round(v);
  }
  function dangerOf(t, threats, seenAll, hand) {
    let mx = 0;
    for (const th of threats) mx = Math.max(mx, dangerOne(t, th, seenAll, hand));
    return mx;
  }
  function dangerLabel(v) {
    if (v <= 0) return '現物';
    if (v <= 6) return '安全寄り';
    if (v <= 12) return 'やや安全';
    if (v <= 20) return '危険';
    return '超危険';
  }

  // ---------- 打牌候補 (3n+2 枚) ----------
  // opt: {melds, seen, threats, winds:{round,seat}}
  function isolationScore(c, t) {
    // 小さいほど不要な牌 (同シャンテン・同受け入れ時のタイブレーク)
    if (t >= 27) return c[t] >= 2 ? 6 : 0;
    const n = t % 9, b = t - n;
    let v = [1, 2, 3, 4, 4, 4, 3, 2, 1][n];
    for (let k = -2; k <= 2; k++) {
      if (!k || n + k < 0 || n + k > 8) continue;
      if (c[b + n + k]) v += Math.abs(k) === 1 ? 3 : 1.5;
    }
    if (c[t] >= 2) v += 4;
    return v;
  }
  function analyzeDiscard(c, opt) {
    const melds = opt.melds || 0, seen = opt.seen || new Int8Array(34), threats = opt.threats || [], dora = opt.dora || null;
    const cur = shanten(c, melds);
    const cands = [];
    for (let t = 0; t < 34; t++) {
      if (!c[t]) continue;
      c[t]--;
      const s = shanten(c, melds);
      const u = ukeire(c, melds, seen, s);
      const iso = isolationScore(c, t);
      c[t]++;
      const isDora = dora ? dora[t] : 0;
      let danger = threats.length ? dangerOf(t, threats, seen, c) : null;
      if (danger && isDora) danger = Math.min(40, Math.round(danger * 1.25));
      // ドラは受け入れ5枚ぶんの価値があるとみなして残す
      cands.push({ t, shanten: s, ukeire: u.total, score: u.total - 5 * isDora, waits: u.tiles, iso, danger, dora: isDora });
    }
    const eff = (a, b) => a.shanten - b.shanten || b.score - a.score || a.iso - b.iso || (a.danger || 0) - (b.danger || 0);
    cands.sort(eff);
    const best = cands[0];
    let mode = 'efficiency', pick = best, note = '';
    if (cur === -1) {
      mode = 'agari';
      note = '和了形です(ツモ和了できるか役を確認)';
    } else if (threats.length) {
      const safest = cands.slice().sort((a, b) => a.danger - b.danger || eff(a, b))[0];
      if (best.shanten >= 2) {
        mode = 'fold'; pick = safest;
        note = '手が遠い(' + best.shanten + '向聴)のでオリ推奨';
      } else {
        // 押し: 最良シャンテンを維持し受け入れ7割以上の中で最も安全な牌
        const ok = cands.filter((x) => x.shanten === best.shanten && x.ukeire >= best.ukeire * 0.7);
        ok.sort((a, b) => a.danger - b.danger || eff(a, b));
        pick = ok[0];
        mode = 'push';
        note = (best.shanten === 0 ? '聴牌' : '1向聴') + ' → 押し候補。切る牌の危険度 ' + pick.danger +
          (pick.danger >= 18 ? '。厳しければ ' + label(safest.t) + '(危険度' + safest.danger + ')でオリ' : '');
        pick = Object.assign({}, pick, { alt: safest.t !== pick.t ? safest : null });
      }
    }
    return { shanten: cur, cands, pick, mode, note, doraCount: countDora(c, dora) };
  }

  // ---------- 鳴き判断 (3n+1 枚) ----------
  function yakuHint(c, melds, winds) {
    // 鳴いても役が付きそうかの簡易判定
    const hints = [];
    const yakuhai = [31, 32, 33];
    if (winds) { yakuhai.push(27 + winds.round); if (winds.seat !== winds.round) yakuhai.push(27 + winds.seat); }
    let total = 0, yao = 0, suits = [0, 0, 0], honors = 0, toi = 0;
    for (let t = 0; t < 34; t++) {
      if (!c[t]) continue;
      total += c[t];
      if (t >= 27) honors += c[t]; else suits[(t / 9) | 0] += c[t];
      if (t >= 27 || t % 9 === 0 || t % 9 === 8) yao += c[t];
      if (c[t] >= 2) toi++;
    }
    for (const y of yakuhai) if (c[y] >= 2) hints.push('役牌(' + label(y) + ')');
    if (yao <= 1) hints.push('タンヤオ');
    const mx = Math.max(suits[0], suits[1], suits[2]);
    if (total >= 4 && mx + honors >= total - 1) hints.push(honors ? '混一色' : '清一色');
    if (toi >= Math.max(3, 5 - melds - 1)) hints.push('対々和');
    return hints;
  }
  // 各「鳴ける牌」について 鳴く/スルー を判定する。
  // 戻り値 calls: [{t, kind:'pon'|'chi', use, shanten, ukeire, discard, hints, verdict:'call'|'maybe'|'skip', reason}]
  function analyzeCalls(c, opt) {
    const melds = opt.melds || 0, seen = opt.seen || new Int8Array(34), dora = opt.dora || null;
    const s0 = shanten(c, melds);
    const u0 = ukeire(c, melds, seen, s0);
    const out = { shanten: s0, ukeire: u0.total, waits: u0.tiles, calls: [], doraCount: countDora(c, dora) };
    if (s0 <= 0) return out; // 聴牌時は待ちを表示 (鳴いて待ちを変える判断は扱わない)
    const yakuhai = [31, 32, 33, 27 + (opt.winds ? opt.winds.round : 0)];
    if (opt.winds) yakuhai.push(27 + opt.winds.seat);
    const isYao = (x) => x >= 27 || x % 9 === 0 || x % 9 === 8;
    const tryCall = (t, use, kind) => {
      for (const x of use) c[x]--;
      let bestS = 9, bestU = -1, bestD = -1;
      for (let d = 0; d < 34; d++) {
        if (!c[d]) continue;
        if (kind === 'chi' && d === t) continue; // 喰い替え(現物)は不可
        c[d]--;
        const s = shanten(c, melds + 1);
        let u = -1;
        if (s <= bestS) u = ukeire(c, melds + 1, seen, s).total - (dora && dora[d] ? 5 : 0);
        c[d]++;
        if (s < bestS || (s === bestS && u > bestU)) { bestS = s; bestU = u; bestD = d; }
      }
      let hints = [];
      if (bestD >= 0) {
        c[bestD]--;
        hints = yakuHint(c, melds + 1, opt.winds);
        bestU = ukeire(c, melds + 1, seen, bestS).total;
        c[bestD]++;
        if (kind === 'pon' && yakuhai.includes(t)) { hints = hints.filter((h) => !h.includes('(' + label(t) + ')')); hints.unshift('役牌(' + label(t) + ')確定'); }
        if ([t].concat(use).some(isYao)) hints = hints.filter((h) => h !== 'タンヤオ');
      }
      for (const x of use) c[x]++;
      const dn = countDora(c, dora) + (dora ? dora[t] : 0);
      const sure = kind === 'pon' && yakuhai.includes(t);
      const strong = sure || hints.some((h) => /混一色|清一色|対々和/.test(h));
      let verdict, reason;
      if (sure && bestS === s0 + 1 && !melds && shantenChiitoi(c) === s0 && shantenNormal(c, 0) > s0) { verdict = 'maybe'; reason = '七対子を見切って役牌で進める手もある'; }
      else if (bestS >= s0) { verdict = 'skip'; reason = (!melds && shantenChiitoi(c) === s0 && shantenNormal(c, 0) > s0) ? '七対子' + s0 + '向聴を崩すだけで進まない' : '向聴数が進まない'; }
      else if (!hints.length && !melds) { verdict = 'skip'; reason = '鳴くと役がなくなる'; }
      else if (sure) { verdict = 'call'; reason = '役牌が確定して手が進む'; }
      else if (melds) { verdict = 'call'; reason = '副露済みなので速度優先'; }
      else if (!strong && ((s0 <= 1 && u0.total >= 12) || (s0 === 2 && u0.total >= 30))) { verdict = 'skip'; reason = '門前のままリーチを狙える'; }
      else if (!strong && hints.every((h) => h.startsWith('役牌(')) ) { verdict = 'maybe'; reason = '役牌がまだ対子(後付けになる)'; }
      else if (!strong && dn === 0) { verdict = 'maybe'; reason = 'ドラなしで安い。急ぐ局面なら鳴く'; }
      else { verdict = 'call'; reason = bestS === 0 ? '聴牌が取れる' : dn ? 'ドラ' + dn + '枚で打点あり' : '役が見えて手が進む'; }
      out.calls.push({ t, kind, use, shanten: bestS, ukeire: bestU, discard: bestD, hints, verdict, reason, good: verdict === 'call' });
    };
    for (let t = 0; t < 34; t++) {
      if (4 - c[t] - seen[t] <= 0) continue;
      if (c[t] >= 2) tryCall(t, [t, t], 'pon');
      if (t < 27) {
        const n = t % 9;
        if (n >= 2 && c[t - 2] && c[t - 1]) tryCall(t, [t - 2, t - 1], 'chi');
        if (n >= 1 && n <= 7 && c[t - 1] && c[t + 1]) tryCall(t, [t - 1, t + 1], 'chi');
        if (n <= 6 && c[t + 1] && c[t + 2]) tryCall(t, [t + 1, t + 2], 'chi');
      }
    }
    // 同じ牌・同じ種類は最良の1つに
    const rank = { call: 0, maybe: 1, skip: 2 };
    const better = (a, b) => rank[a.verdict] - rank[b.verdict] || a.shanten - b.shanten || b.ukeire - a.ukeire;
    const byKey = new Map();
    for (const k of out.calls) {
      const key = k.t + k.kind, p = byKey.get(key);
      if (!p || better(k, p) < 0) byKey.set(key, k);
    }
    out.calls = Array.from(byKey.values()).sort((a, b) => better(a, b) || a.t - b.t);
    return out;
  }

  // 文字列 "123m456p77z" → counts
  function parse(str) {
    const c = new Int8Array(34);
    let buf = [];
    for (const ch of str) {
      if (ch >= '0' && ch <= '9') buf.push(+ch);
      else {
        const b = { m: 0, p: 9, s: 18, z: 27 }[ch];
        for (const n of buf) c[b + (n === 0 ? 5 : n) - 1]++;
        buf = [];
      }
    }
    return c;
  }
  function toStr(c) {
    let s = '';
    for (let g = 0; g < 4; g++) {
      let part = '';
      for (let i = 0; i < (g === 3 ? 7 : 9); i++) part += String(i + 1).repeat(c[g * 9 + i]);
      if (part) s += part + 'mpsz'[g];
    }
    return s;
  }

  const api = { NAMES, label, doraOf, shanten, shantenNormal, ukeire, analyzeDiscard, analyzeCalls, dangerOf, dangerLabel, parse, toStr };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MJ = api;
})(typeof self !== 'undefined' ? self : this);
