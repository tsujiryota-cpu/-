#!/usr/bin/env python3
"""経費精算.xlsx に「出納帳_写真N」シートを追記する。

既存ブックの ZIP エントリはそのままコピーし、新しいワークシートだけを
追加するため、埋め込み画像・セルのコメント・印刷設定などが失われない
（openpyxl で読み書きすると埋め込み画像が消える）。

セル書式は既存シートと同じ styles.xml の xf インデックスを流用する:
  s=9  タイトル(太字14pt・中央)   s=10 見出し(青背景・太字・罫線)
  s=11 日付(m/d/yyyy)            s=12 文字列
  s=13 金額(#,##0)               s=14 合計ラベル(橙背景)
  s=15 合計金額(橙背景・#,##0)    s=3  空白セル(罫線のみ)

使い方:
  python3 add_sheet.py --workbook 経費精算.xlsx --rows rows.json --out out.xlsx

rows.json:
  {
    "title": "出納帳（写真4：2026年8月12日〜8月14日）",   // 省略時は日付から自動生成
    "sheet_name": "出納帳_写真4",                          // 省略時は自動連番
    "rows": [
      {"date": "2026-08-12", "payee": "支払先", "description": "内容",
       "method": "現金", "amount": 1234}
    ]
  }
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
import unicodedata
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

WS_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"
)
WS_REL_TYPE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
)
HEADERS = ["日付", "支払先", "内容", "支払方法", "金額（円）"]
EXCEL_EPOCH = dt.date(1899, 12, 30)  # Excel 1900 日付システムの基準日

# 既存シートから流用する書式インデックス
S_TITLE, S_HEADER, S_DATE, S_TEXT, S_AMOUNT = 9, 10, 11, 12, 13
S_TOTAL_LABEL, S_TOTAL_VALUE, S_BLANK = 14, 15, 3


def display_width(text: str) -> float:
    """全角を2、半角を1として文字列の表示幅を返す。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WFA" else 1 for ch in text)


def parse_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y年%m月%d日"):
        try:
            return dt.datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    raise SystemExit(f"日付を解釈できません: {value!r}（例: 2026-08-12）")


def parse_amount(value) -> int:
    if isinstance(value, (int, float)):
        return int(round(value))
    text = re.sub(r"[,，\s円¥￥]", "", str(value))
    if not re.fullmatch(r"-?\d+", text):
        raise SystemExit(f"金額を解釈できません: {value!r}（例: 1234）")
    return int(text)


def serial(date: dt.date) -> int:
    return (date - EXCEL_EPOCH).days


def next_sheet_number(names: list[str]) -> int:
    used = [int(m.group(1)) for n in names if (m := re.search(r"写真(\d+)$", n))]
    return max(used, default=0) + 1


def make_title(number: int, dates: list[dt.date]) -> str:
    start, end = min(dates), max(dates)
    if start == end:
        return f"出納帳（写真{number}：{start.year}年{start.month}月{start.day}日）"
    span = f"{start.year}年{start.month}月{start.day}日〜"
    span += (
        f"{end.month}月{end.day}日"
        if start.year == end.year
        else f"{end.year}年{end.month}月{end.day}日"
    )
    return f"出納帳（写真{number}：{span}）"


def inline(ref: str, style: int, text: str) -> str:
    return (
        f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">'
        f"{escape(text)}</t></is></c>"
    )


def build_sheet_xml(title: str, rows: list[dict]) -> str:
    first, last = 3, 2 + len(rows)
    total_row = last + 1

    # 列幅: Excel の bestFit に近づけるため表示幅 + 余白
    widths = [10.2]
    for col, header in (("payee", HEADERS[1]), ("description", HEADERS[2]), ("method", HEADERS[3])):
        longest = max([display_width(str(r[col])) for r in rows] + [display_width(header)])
        widths.append(round(min(max(longest + 0.7, 9.0), 70.0), 8))
    widths.append(10.4)

    out = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        f'<dimension ref="A1:E{total_row}"/>',
        "<sheetViews><sheetView workbookViewId=\"0\"/></sheetViews>",
        '<sheetFormatPr defaultRowHeight="18"/>',
        "<cols>"
        + "".join(
            f'<col min="{i}" max="{i}" width="{w}" bestFit="1" customWidth="1"/>'
            for i, w in enumerate(widths, start=1)
        )
        + "</cols>",
        "<sheetData>",
        f'<row r="1" spans="1:5" ht="22.2">{inline("A1", S_TITLE, title)}'
        + "".join(f'<c r="{c}1" s="{S_TITLE}"/>' for c in "BCDE")
        + "</row>",
        f'<row r="2" spans="1:5">'
        + "".join(
            inline(f"{c}2", S_HEADER, h) for c, h in zip("ABCDE", HEADERS)
        )
        + "</row>",
    ]

    for offset, row in enumerate(rows):
        r = first + offset
        out.append(
            f'<row r="{r}" spans="1:5">'
            f'<c r="A{r}" s="{S_DATE}"><v>{serial(row["date"])}</v></c>'
            + inline(f"B{r}", S_TEXT, str(row["payee"]))
            + inline(f"C{r}", S_TEXT, str(row["description"]))
            + inline(f"D{r}", S_TEXT, str(row["method"]))
            + f'<c r="E{r}" s="{S_AMOUNT}"><v>{row["amount"]}</v></c>'
            "</row>"
        )

    out.append(
        f'<row r="{total_row}" spans="1:5">'
        + "".join(f'<c r="{c}{total_row}" s="{S_BLANK}"/>' for c in "ABC")
        + inline(f"D{total_row}", S_TOTAL_LABEL, "合計")
        + f'<c r="E{total_row}" s="{S_TOTAL_VALUE}">'
        f"<f>SUM(E{first}:E{last})</f></c>"
        "</row>"
    )
    out += [
        "</sheetData>",
        '<mergeCells count="1"><mergeCell ref="A1:E1"/></mergeCells>',
        '<phoneticPr fontId="2"/>',
        '<pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75"'
        ' header="0.3" footer="0.3"/>',
        "</worksheet>",
    ]
    return "".join(out)


def add_sheet(src: Path, dst: Path, spec: dict) -> tuple[str, int, int]:
    rows = spec.get("rows") or []
    if not rows:
        raise SystemExit("rows が空です。1件以上の明細が必要です。")
    for row in rows:
        missing = [k for k in ("date", "payee", "description", "method", "amount") if k not in row]
        if missing:
            raise SystemExit(f"明細に不足キーがあります: {missing} -> {row}")
        row["date"] = parse_date(row["date"])
        row["amount"] = parse_amount(row["amount"])
    rows.sort(key=lambda r: r["date"])

    with zipfile.ZipFile(src) as zin:
        names = zin.namelist()
        workbook = zin.read("xl/workbook.xml").decode("utf-8")
        rels = zin.read("xl/_rels/workbook.xml.rels").decode("utf-8")
        content_types = zin.read("[Content_Types].xml").decode("utf-8")
        entries = [(i, zin.read(i.filename)) for i in zin.infolist()]

    existing = re.findall(r'<sheet name="([^"]+)"', workbook)
    number = next_sheet_number(existing)
    sheet_name = spec.get("sheet_name") or f"出納帳_写真{number}"
    if sheet_name in existing:
        raise SystemExit(f"シート名が既に存在します: {sheet_name}")
    title = spec.get("title") or make_title(number, [r["date"] for r in rows])

    used_ids = {int(m) for m in re.findall(r'Id="rId(\d+)"', rels)}
    rel_id = f"rId{max(used_ids) + 1}"
    used_files = {int(m) for m in re.findall(r"worksheets/sheet(\d+)\.xml", " ".join(names))}
    part = f"xl/worksheets/sheet{max(used_files) + 1}.xml"
    sheet_ids = {int(m) for m in re.findall(r'sheetId="(\d+)"', workbook)}
    sheet_id = max(sheet_ids) + 1

    workbook = workbook.replace(
        "</sheets>",
        f'<sheet name="{escape(sheet_name)}" sheetId="{sheet_id}" r:id="{rel_id}"/></sheets>',
    )
    rels = rels.replace(
        "</Relationships>",
        f'<Relationship Id="{rel_id}" Type="{WS_REL_TYPE}"'
        f' Target="{part[len("xl/"):]}"/></Relationships>',
    )
    content_types = content_types.replace(
        "</Types>",
        f'<Override PartName="/{part}" ContentType="{WS_CONTENT_TYPE}"/></Types>',
    )
    # calcChain は追加した SUM を含まないので破棄する（Excel が再構築する）
    rels = re.sub(r'<Relationship[^>]*Target="calcChain\.xml"\s*/>', "", rels)
    content_types = re.sub(r'<Override PartName="/xl/calcChain\.xml"[^>]*/>', "", content_types)

    replacements = {
        "xl/workbook.xml": workbook.encode("utf-8"),
        "xl/_rels/workbook.xml.rels": rels.encode("utf-8"),
        "[Content_Types].xml": content_types.encode("utf-8"),
    }
    sheet_xml = build_sheet_xml(title, rows).encode("utf-8")

    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for info, data in entries:
            if info.filename == "xl/calcChain.xml":
                continue
            zout.writestr(info, replacements.get(info.filename, data))
        zout.writestr(part, sheet_xml)

    return sheet_name, len(rows), sum(r["amount"] for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="経費精算.xlsx に出納帳シートを追記する")
    ap.add_argument("--workbook", required=True, type=Path, help="既存の経費精算.xlsx")
    ap.add_argument("--rows", required=True, type=Path, help="明細 JSON")
    ap.add_argument("--out", type=Path, help="出力先（省略時は --workbook を上書き）")
    args = ap.parse_args()

    spec = json.loads(args.rows.read_text(encoding="utf-8"))
    dst = args.out or args.workbook
    tmp = dst.with_name(dst.stem + ".tmp.xlsx")
    name, count, total = add_sheet(args.workbook, tmp, spec)

    # 保存後に読み込み検証（破損していれば例外になる）
    try:
        import openpyxl

        wb = openpyxl.load_workbook(tmp)
        if name not in wb.sheetnames:
            raise SystemExit("検証失敗: 追加したシートが見つかりません")
    except ImportError:
        print("警告: openpyxl 未インストールのため検証を省略しました", file=sys.stderr)

    shutil.move(tmp, dst)
    print(f"追加: {name}（{count}件 / 合計 {total:,}円） -> {dst}")


if __name__ == "__main__":
    main()
