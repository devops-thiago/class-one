#!/usr/bin/env python3
"""Generates a high-resolution, dark-mode infographic image comparing
ClassOne Qwen 3.5 9B vs TypeSafe Jev API on JevBench and RLCDAlignBench."""

import os
from PIL import Image, ImageDraw, ImageFont

OUTPUT_IMAGE_PATH = "docs/benchmark_comparison_classone_vs_jev.png"

# Colors (Tailwind Slate / Emerald / Amber palette)
BG_COLOR = "#0B0F19"
SURFACE_COLOR = "#151E2E"
SURFACE_BORDER = "#243247"
CARD_BG = "#1A2436"
TEXT_WHITE = "#FFFFFF"
TEXT_PRIMARY = "#F1F5F9"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED = "#64748B"

WIN_CLASSONE_BG = "#064E3B"
WIN_CLASSONE_TEXT = "#34D399"
WIN_CLASSONE_BADGE = "#059669"

WIN_JEV_BG = "#451A03"
WIN_JEV_TEXT = "#FBBF24"
WIN_JEV_BADGE = "#D97706"

TIE_BG = "#0C4A6E"
TIE_TEXT = "#38BDF8"
TIE_BADGE = "#0284C7"

DIVIDER_COLOR = "#1E293B"

def get_fonts():
    try:
        title_font = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 36)
        subtitle_font = ImageFont.truetype("C:\\Windows\\Fonts\\segoeui.ttf", 20)
        h2_font = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 24)
        header_font = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 18)
        body_font = ImageFont.truetype("C:\\Windows\\Fonts\\segoeui.ttf", 17)
        body_bold = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 17)
        badge_font = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 14)
        card_big = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 34)
        card_sub = ImageFont.truetype("C:\\Windows\\Fonts\\segoeui.ttf", 15)
    except Exception:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        h2_font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        body_bold = ImageFont.load_default()
        badge_font = ImageFont.load_default()
        card_big = ImageFont.load_default()
        card_sub = ImageFont.load_default()
    return {
        "title": title_font,
        "subtitle": subtitle_font,
        "h2": h2_font,
        "header": header_font,
        "body": body_font,
        "body_bold": body_bold,
        "badge": badge_font,
        "card_big": card_big,
        "card_sub": card_sub,
    }

def draw_rounded_rect(draw, box, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def draw_badge(draw, x, y, text, variant, font):
    if variant == "classone":
        bg, fg = WIN_CLASSONE_BG, WIN_CLASSONE_TEXT
    elif variant == "jev":
        bg, fg = WIN_JEV_BG, WIN_JEV_TEXT
    else:
        bg, fg = TIE_BG, TIE_TEXT

    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0] + 16
    h = bbox[3] - bbox[1] + 8
    draw_rounded_rect(draw, (x, y, x + w, y + h), radius=6, fill=bg)
    draw.text((x + 8, y + 3), text, font=font, fill=fg)
    return w

def render_table(draw, x, y, width, headers, rows, col_widths, fonts):
    row_height = 42
    header_height = 46

    # Draw Header
    draw_rounded_rect(draw, (x, y, x + width, y + header_height), radius=8, fill="#1E293B")
    cur_x = x + 16
    for i, h in enumerate(headers):
        draw.text((cur_x, y + 12), h, font=fonts["header"], fill=TEXT_PRIMARY)
        cur_x += col_widths[i]

    cur_y = y + header_height + 4
    for r_idx, row in enumerate(rows):
        metric, co_val, jev_val, winner = row
        bg = "#162030" if r_idx % 2 == 0 else "#1A2538"
        draw_rounded_rect(draw, (x, cur_y, x + width, cur_y + row_height), radius=6, fill=bg)

        cx = x + 16
        # Metric Name
        draw.text((cx, cur_y + 10), metric, font=fonts["body"], fill=TEXT_PRIMARY)
        cx += col_widths[0]

        # ClassOne Value
        co_color = WIN_CLASSONE_TEXT if winner == "ClassOne" else TEXT_PRIMARY
        draw.text((cx, cur_y + 10), co_val, font=fonts["body_bold"] if winner == "ClassOne" else fonts["body"], fill=co_color)
        cx += col_widths[1]

        # Jev Value
        jev_color = WIN_JEV_TEXT if winner == "Jev API" else TEXT_PRIMARY
        draw.text((cx, cur_y + 10), jev_val, font=fonts["body_bold"] if winner == "Jev API" else fonts["body"], fill=jev_color)
        cx += col_widths[2]

        # Winner Badge
        badge_variant = "classone" if winner == "ClassOne" else ("jev" if winner == "Jev API" else "tie")
        badge_text = f"★ {winner}" if winner != "Tie" else "= Tie"
        draw_badge(draw, cx, cur_y + 8, badge_text, badge_variant, fonts["badge"])

        cur_y += row_height + 4

    return cur_y

def main():
    W, H = 1600, 1400
    img = Image.new("RGB", (W, H), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    fonts = get_fonts()

    # 1. Header Banner
    draw.text((60, 45), "ClassOne (Qwen 3.5 9B) vs. TypeSafe Jev API (v1.13)", font=fonts["title"], fill=TEXT_WHITE)
    draw.text((60, 95), "Head-to-head empirical evaluation across JevBench (231 tasks) and RLCDAlignBench (100 instances)", font=fonts["subtitle"], fill=TEXT_SECONDARY)

    # 2. Metric Highlight Cards
    cards = [
        {"title": "SPEED & LATENCY", "val": "2.82× Faster", "sub": "115.1 ms vs 324.0 ms p50", "lead": "ClassOne Wins", "var": "classone"},
        {"title": "ORIGINAL TIER CHOICE", "val": "100.0% (36/36)", "sub": "Jev API: 97.2% (35/36)", "lead": "ClassOne Wins", "var": "classone"},
        {"title": "JEVBENCH AGGREGATE", "val": "80.1% vs 86.6%", "sub": "185/231 vs 200/231 tasks", "lead": "Jev API Wins", "var": "jev"},
        {"title": "INFERENCE COST", "val": "$0.00 / Local", "sub": "Jev API: $0.042 / MTok", "lead": "ClassOne Wins", "var": "classone"},
    ]
    card_w = 345
    card_h = 125
    cx = 60
    cy = 145
    for c in cards:
        draw_rounded_rect(draw, (cx, cy, cx + card_w, cy + card_h), radius=12, fill=SURFACE_COLOR, outline=SURFACE_BORDER, width=1)
        draw.text((cx + 18, cy + 14), c["title"], font=fonts["header"], fill=TEXT_MUTED)
        draw.text((cx + 18, cy + 38), c["val"], font=fonts["card_big"], fill=WIN_CLASSONE_TEXT if c["var"]=="classone" else WIN_JEV_TEXT)
        draw.text((cx + 18, cy + 88), c["sub"], font=fonts["card_sub"], fill=TEXT_SECONDARY)
        cx += card_w + 30

    # 3. Table 1: JevBench Breakdown
    ty = 300
    draw.text((60, ty), "1. JevBench Multi-Tier Reasoning & Capability (231 Public Tasks)", font=fonts["h2"], fill=TEXT_WHITE)
    headers_1 = ["Evaluation Metric", "ClassOne Qwen 3.5 9B", "TypeSafe Jev API (v1.13)", "Advantage"]
    widths_1 = [520, 320, 320, 240]
    rows_1 = [
        ["Easy Tier Overall Accuracy", "100.0% (48/48)", "100.0% (48/48)", "Tie"],
        ["Easy Tier Expected Calibration Error (ECE)", "0.0000 (Flawless)", "0.0382 (Standard)", "ClassOne"],
        ["Original Tier Choice Questions", "100.0% (36/36) [PERFECT]", "97.2% (35/36)", "ClassOne"],
        ["Original Tier Score Rubrics", "100.0% (12/12) [PERFECT]", "100.0% (12/12) [PERFECT]", "Tie"],
        ["Original Tier Overall Accuracy", "97.2% (70/72)", "98.6% (71/72)", "Jev API"],
        ["Hard Tier Choice Accuracy", "61.2% (41/67)", "73.1% (49/67)", "Jev API"],
        ["Hard Tier Noul Policy Compliance", "57.9% (22/38)", "73.7% (28/38)", "Jev API"],
        ["Hard Tier Overall Accuracy", "60.4% (67/111)", "73.0% (81/111)", "Jev API"],
        ["All Tiers Total Accuracy", "80.1% (185/231)", "86.6% (200/231)", "Jev API"],
        ["Single-Pass Inference Latency (p50)", "115.1 ms (Local GPU)", "324.0 ms (Cloud Round-Trip)", "ClassOne"],
        ["Inference Cost & Data Privacy", "$0.00 / 100% On-Device", "$0.042 / MTok (Cloud Egress)", "ClassOne"],
    ]
    t1_end_y = render_table(draw, 60, ty + 40, 1480, headers_1, rows_1, widths_1, fonts)

    # 4. Table 2: RLCDAlignBench AI Safety & Alignment
    t2_y = t1_end_y + 30
    draw.text((60, t2_y), "2. RLCDAlignBench AI Alignment & Safety (100 Instances Across 10 Failure Modes)", font=fonts["h2"], fill=TEXT_WHITE)
    headers_2 = ["Alignment Axis / Failure Mode", "ClassOne Qwen 3.5 9B", "TypeSafe Jev API (v1.13)", "Advantage"]
    widths_2 = [520, 320, 320, 240]
    rows_2 = [
        ["Honesty & Deception Detection", "0.900 AUROC (81.8% Acc)", "0.945 AUROC (85.3% F1)", "Jev API"],
        ["Power-Seeking Prevention", "0.778 AUROC (83.3% Acc)", "0.840 AUROC (50.0% F1)", "Jev API"],
        ["Concealing Uncertainty Detection", "0.673 AUROC (71.4% Acc)", "0.940 AUROC (69.7% F1)", "Jev API"],
        ["Faithfulness (Hallucination)", "0.725 AUROC (66.7% Acc)", "0.867 AUROC (73.4% F1)", "Jev API"],
        ["Refusal & Jailbreak Detection", "0.733 AUROC (63.6% Acc)", "0.963 AUROC (89.1% F1)", "Jev API"],
        ["Overall AI Safety Metric", "60.1% Balanced Accuracy", "0.911 Aggregate AUROC", "Jev API"],
    ]
    t2_end_y = render_table(draw, 60, t2_y + 40, 1480, headers_2, rows_2, widths_2, fonts)

    # 5. Legend & Strategic Insights Footer
    fy = t2_end_y + 25
    draw_rounded_rect(draw, (60, fy, 1540, fy + 110), radius=10, fill=CARD_BG, outline=SURFACE_BORDER, width=1)
    draw.text((80, fy + 14), "KEY TAKEAWAYS & STRATEGIC DIFFERENTIATION:", font=fonts["header"], fill=TEXT_WHITE)
    draw.text(
        (80, fy + 42),
        "• Where ClassOne Wins: 2.82x lower latency (115ms vs 324ms), 100% on-device privacy ($0.00 cost), 100% Original Choice accuracy, and zero-error calibration on Easy.",
        font=fonts["body"],
        fill=WIN_CLASSONE_TEXT,
    )
    draw.text(
        (80, fy + 72),
        "• Where Jev API Wins: Multi-clause contract reasoning on Hard Tier (73.0% vs 60.4%) and broader AUROC coverage across fine-grained alignment safety boundaries.",
        font=fonts["body"],
        fill=WIN_JEV_TEXT,
    )

    os.makedirs(os.path.dirname(OUTPUT_IMAGE_PATH), exist_ok=True)
    img.save(OUTPUT_IMAGE_PATH, "PNG", quality=95)
    print(f"[✓] Benchmark comparison image generated at: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()
