#!/usr/bin/env python3
"""Generates a clean 3-column comparison table image:
Criteria | My Model | Jev
Colors winning (larger) cells with green, with no additional info."""

import os

from PIL import Image, ImageDraw, ImageFont

OUTPUT_PATH = "docs/benchmark_comparison_classone_vs_jev.png"

# Color definitions
BG_COLOR = "#0B0F19"
TABLE_BORDER = "#2D3748"
HEADER_BG = "#1A202C"
CELL_DEFAULT_BG = "#171923"
CELL_GREEN_BG = "#15803D"
TEXT_WHITE = "#FFFFFF"
TEXT_MUTED = "#CBD5E0"

# Table data: (Criteria, My Model Value, Jev Value, My Numeric, Jev Numeric)
DATA = [
    ("Easy Tier Accuracy", "100.0%", "100.0%", 100.0, 100.0),
    ("Original Tier Choice", "100.0%", "97.2%", 100.0, 97.2),
    ("Original Tier Score", "100.0%", "100.0%", 100.0, 100.0),
    ("Original Tier Accuracy", "97.2%", "98.6%", 97.2, 98.6),
    ("Hard Tier Choice", "61.2%", "73.1%", 61.2, 73.1),
    ("Hard Tier Noul", "57.9%", "73.7%", 57.9, 73.7),
    ("Hard Tier Accuracy", "60.4%", "73.0%", 60.4, 73.0),
    ("JevBench Overall", "80.1%", "86.6%", 80.1, 86.6),
    ("Inference Speed (req/s)", "8.7", "3.1", 8.7, 3.1),
    ("Honesty AUROC", "0.900", "0.945", 0.900, 0.945),
    ("Power Seeking AUROC", "0.778", "0.840", 0.778, 0.840),
    ("Faithfulness AUROC", "0.725", "0.867", 0.725, 0.867),
    ("Refusal AUROC", "0.733", "0.963", 0.733, 0.963),
    ("AlignBench Overall AUROC", "0.594", "0.911", 0.594, 0.911),
]


def get_fonts():
    try:
        header_font = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 20)
        body_font = ImageFont.truetype("C:\\Windows\\Fonts\\segoeui.ttf", 18)
        body_bold = ImageFont.truetype("C:\\Windows\\Fonts\\segoeuib.ttf", 18)
    except Exception:
        header_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        body_bold = ImageFont.load_default()
    return header_font, body_font, body_bold


def main():
    col_w = [400, 290, 220]
    total_w = sum(col_w)
    row_h = 44
    header_h = 50
    margin_x = 40
    margin_y = 40

    img_w = total_w + margin_x * 2
    img_h = header_h + len(DATA) * row_h + margin_y * 2

    img = Image.new("RGB", (img_w, img_h), color=BG_COLOR)
    draw = ImageDraw.Draw(img)
    header_font, body_font, body_bold = get_fonts()

    # Draw Header Row
    headers = ["Criteria", "ClassOne Qwen 3.5 9B", "Jev"]
    x = margin_x
    y = margin_y

    for i, h in enumerate(headers):
        w = col_w[i]
        draw.rectangle([x, y, x + w, y + header_h], fill=HEADER_BG, outline=TABLE_BORDER, width=1)
        # Center text
        bbox = header_font.getbbox(h)
        tx = x + (w - (bbox[2] - bbox[0])) // 2 if i > 0 else x + 20
        ty = y + (header_h - (bbox[3] - bbox[1])) // 2
        draw.text((tx, ty), h, font=header_font, fill=TEXT_WHITE)
        x += w

    # Draw Data Rows
    y += header_h
    for crit, my_str, jev_str, my_val, jev_val in DATA:
        x = margin_x
        my_wins = my_val > jev_val
        jev_wins = jev_val > my_val

        # Column 1: Criteria
        w1 = col_w[0]
        draw.rectangle([x, y, x + w1, y + row_h], fill=CELL_DEFAULT_BG, outline=TABLE_BORDER, width=1)
        bbox1 = body_font.getbbox(crit)
        ty = y + (row_h - (bbox1[3] - bbox1[1])) // 2
        draw.text((x + 20, ty), crit, font=body_font, fill=TEXT_MUTED)
        x += w1

        # Column 2: My Model
        w2 = col_w[1]
        fill2 = CELL_GREEN_BG if my_wins else CELL_DEFAULT_BG
        draw.rectangle([x, y, x + w2, y + row_h], fill=fill2, outline=TABLE_BORDER, width=1)
        font2 = body_bold if my_wins else body_font
        bbox2 = font2.getbbox(my_str)
        tx2 = x + (w2 - (bbox2[2] - bbox2[0])) // 2
        draw.text((tx2, ty), my_str, font=font2, fill=TEXT_WHITE)
        x += w2

        # Column 3: Jev
        w3 = col_w[2]
        fill3 = CELL_GREEN_BG if jev_wins else CELL_DEFAULT_BG
        draw.rectangle([x, y, x + w3, y + row_h], fill=fill3, outline=TABLE_BORDER, width=1)
        font3 = body_bold if jev_wins else body_font
        bbox3 = font3.getbbox(jev_str)
        tx3 = x + (w3 - (bbox3[2] - bbox3[0])) // 2
        draw.text((tx3, ty), jev_str, font=font3, fill=TEXT_WHITE)

        y += row_h

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    img.save(OUTPUT_PATH, "PNG", quality=95)
    print(f"[✓] Successfully generated clean table image at: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
