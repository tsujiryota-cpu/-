#!/usr/bin/env python3
"""Slack用カスタム絵文字ジェネレータ (128x128 透過PNG)。
- 文字スタンプ: MEGAMOJI風に、文字を正方形いっぱいに引き伸ばして縁取り
- キャラスタンプ: オリジナルキャラ「めぶきち」(葉っぱの生えた丸い生き物)
使い方: python3 make.py   → このフォルダに PNG を出力
"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.dirname(os.path.abspath(__file__))
FONT = "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf"
S = 512          # 作業解像度 (4倍で描いて縮小)
FINAL = 128


def save(img, name):
    img.resize((FINAL, FINAL), Image.LANCZOS).save(os.path.join(OUT, name + ".png"))


# ---------- 文字スタンプ ----------
def line_image(text, color, outline="#ffffff"):
    """1行を太字+白フチで描き、余白を切り落とした画像を返す"""
    f = ImageFont.truetype(FONT, 300)
    sw, ow = 14, 30   # 太らせ幅 / フチ幅
    w = int(f.getlength(text)) + ow * 4
    im = Image.new("RGBA", (w, 460), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((ow * 2, 60), text, font=f, fill=outline, stroke_width=ow, stroke_fill=outline)
    d.text((ow * 2, 60), text, font=f, fill=color, stroke_width=sw, stroke_fill=color)
    return im.crop(im.getbbox())


def text_emoji(name, lines, colors):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    h = S // len(lines)
    for i, (t, c) in enumerate(zip(lines, colors)):
        li = line_image(t, c)
        # 正方形いっぱいに伸縮 (ただし縦横の歪みは1.5倍まで)
        w, hh = S, h
        r = (w / li.width) / (hh / li.height)
        if r > 1.5:
            w = int(w * 1.5 / r)
        elif r < 1 / 1.5:
            hh = int(hh * 1.5 * r)
        li = li.resize((w, hh), Image.LANCZOS)
        img.alpha_composite(li, ((S - w) // 2, i * h + (h - hh) // 2))
    save(img, name)


# ---------- キャラスタンプ ----------
BODY, BODY_D, LEAF, LEAF_D = "#ffd166", "#c98a1b", "#57c66b", "#2f8f45"
INK, CHEEK = "#4a3320", "#ff8fa3"


def base():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # 葉っぱ(2枚)と茎
    d.line((256, 150, 256, 95), fill=LEAF_D, width=14)
    d.ellipse((150, 40, 262, 110), fill=LEAF, outline=LEAF_D, width=8)
    d.ellipse((250, 20, 372, 96), fill=LEAF, outline=LEAF_D, width=8)
    # 足
    d.ellipse((150, 440, 240, 496), fill=BODY, outline=BODY_D, width=10)
    d.ellipse((272, 440, 362, 496), fill=BODY, outline=BODY_D, width=10)
    # 体(横長のまんじゅう型)
    d.ellipse((40, 130, 472, 480), fill=BODY, outline=BODY_D, width=12)
    # ほっぺ
    d.ellipse((90, 330, 160, 372), fill=CHEEK)
    d.ellipse((352, 330, 422, 372), fill=CHEEK)
    return img, d


def eyes(d, kind):
    for x in (176, 336):
        if kind == "dot":
            d.ellipse((x - 20, 262, x + 20, 310), fill=INK)
            d.ellipse((x - 10, 268, x + 2, 282), fill="white")
        elif kind == "happy":   # ^ ^
            d.arc((x - 30, 262, x + 30, 330), 200, 340, fill=INK, width=14)
        elif kind == "closed":  # ‿ 閉じ目
            d.arc((x - 30, 240, x + 30, 300), 20, 160, fill=INK, width=14)
        elif kind == "wide":
            d.ellipse((x - 30, 250, x + 30, 316), fill="white", outline=INK, width=10)
            d.ellipse((x - 10, 272, x + 10, 296), fill=INK)
        elif kind == "sad":
            d.ellipse((x - 18, 270, x + 18, 312), fill=INK)
            s = 1 if x < 256 else -1
            d.line((x - 34 * s, 250, x + 30 * s, 228), fill=INK, width=12)


def mouth(d, kind):
    if kind == "smile":
        d.arc((216, 300, 296, 370), 20, 160, fill=INK, width=12)
    elif kind == "open":
        d.pieslice((206, 290, 306, 400), 0, 180, fill="#e85d75", outline=INK, width=10)
    elif kind == "o":
        d.ellipse((232, 330, 280, 388), fill="#e85d75", outline=INK, width=10)
    elif kind == "frown":
        d.arc((216, 350, 296, 410), 200, 340, fill=INK, width=12)
    elif kind == "flat":
        d.line((226, 360, 286, 360), fill=INK, width=12)


def heart(d, cx, cy, r, c="#ff4d6d"):
    d.ellipse((cx - r, cy - r, cx, cy), fill=c)
    d.ellipse((cx, cy - r, cx + r, cy), fill=c)
    d.polygon([(cx - r, cy - r * 0.35), (cx + r, cy - r * 0.35), (cx, cy + r)], fill=c)


def star(d, cx, cy, r, c="#ffb703"):
    d.polygon([(cx, cy - r), (cx + r * .28, cy - r * .28), (cx + r, cy), (cx + r * .28, cy + r * .28),
               (cx, cy + r), (cx - r * .28, cy + r * .28), (cx - r, cy), (cx - r * .28, cy - r * .28)], fill=c)


def label(img, text, color):
    """下部に文字帯を重ねる"""
    li = line_image(text, color)
    w = min(S - 20, int(li.width * 150 / li.height))
    li = li.resize((w, 150), Image.LANCZOS)
    img.alpha_composite(li, ((S - w) // 2, S - 152))


def chara(name, eye, mth, text=None, color="#e63946", extra=None):
    img, d = base()
    eyes(d, eye); mouth(d, mth)
    if extra:
        extra(d)
    if text:
        label(img, text, color)
    save(img, name)


def ex_hearts(d):
    heart(d, 80, 110, 46); heart(d, 440, 150, 36); heart(d, 400, 60, 26)

def ex_stars(d):
    star(d, 70, 110, 56); star(d, 450, 120, 44); star(d, 420, 240, 26)

def ex_sweat(d):
    d.polygon([(420, 150), (396, 210), (444, 210)], fill="#5bc0eb")
    d.ellipse((394, 190, 446, 242), fill="#5bc0eb")

def ex_tears(d):
    for x in (176, 336):
        d.rounded_rectangle((x - 16, 310, x + 16, 440), 16, fill="#5bc0eb")

def ex_zzz(d):
    f = ImageFont.truetype(FONT, 110)
    d.text((370, 30), "Z", font=f, fill="#4361ee", stroke_width=8, stroke_fill="#4361ee")
    f2 = ImageFont.truetype(FONT, 70)
    d.text((450, 120), "z", font=f2, fill="#4361ee", stroke_width=6, stroke_fill="#4361ee")

def ex_bang(d):
    d.rounded_rectangle((424, 30, 464, 150), 18, fill="#e63946")
    d.ellipse((424, 168, 464, 208), fill="#e63946")

def ex_tea(d):
    d.rounded_rectangle((350, 380, 450, 470), 16, fill="#8ecae6", outline=INK, width=8)
    d.arc((430, 395, 480, 450), 270, 90, fill=INK, width=8)
    d.arc((370, 320, 400, 372), 90, 270, fill="#adb5bd", width=8)
    d.arc((400, 320, 430, 372), 270, 90, fill="#adb5bd", width=8)


if __name__ == "__main__":
    R, B, G, O, P = "#e63946", "#1d6fd8", "#1a9850", "#f3722c", "#8338ec"
    # 文字スタンプ
    text_emoji("ryoukai", ["了解", "です"], [G, G])
    text_emoji("arigato", ["あり", "がと"], [R, O])
    text_emoji("otsukare", ["おつ", "かれ"], [B, P])
    text_emoji("kakunin", ["確認", "します"], [B, B])
    text_emoji("shouchi", ["承知"], [G])
    text_emoji("lgtm", ["LG", "TM"], [P, R])
    text_emoji("saikou", ["最高"], [O])
    text_emoji("taiouchu", ["対応", "中"], [O, R])
    text_emoji("sumi", ["済"], [R])
    text_emoji("onegai", ["お願い", "します"], [P, P])
    # キャラスタンプ
    chara("mebu_ok", "happy", "open", "OK", G)
    chara("mebu_thanks", "happy", "smile", extra=ex_hearts)
    chara("mebu_otsu", "closed", "smile", extra=ex_tea)
    chara("mebu_sorry", "sad", "frown", extra=ex_sweat)
    chara("mebu_naki", "sad", "open", extra=ex_tears)
    chara("mebu_kira", "dot", "open", extra=ex_stars)
    chara("mebu_nemui", "closed", "flat", extra=ex_zzz)
    chara("mebu_bikkuri", "wide", "o", extra=ex_bang)
    chara("mebu_smile", "dot", "smile")
    chara("mebu_naruhodo", "dot", "o", "なるほど", B)

    # 一覧シート (確認用)
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".png") and f != "preview.png")
    cols = 5
    rows = -(-len(files) // cols)
    sheet = Image.new("RGBA", (cols * 148 + 20, rows * 148 + 20), "#f4f4f4")
    for i, f in enumerate(files):
        im = Image.open(os.path.join(OUT, f))
        sheet.alpha_composite(im, (20 + (i % cols) * 148, 20 + (i // cols) * 148))
    sheet.save(os.path.join(OUT, "preview.png"))
