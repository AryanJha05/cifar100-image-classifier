"""
Generate a diverse, self-contained OOD (Out-of-Distribution) validation dataset.
Creates realistic test cases across 5 major categories:
1. Documents / Text (invoices, code screenshots, text sheets)
2. UI Screenshots (app dialogs, control panels, web bars)
3. Logos / Symbols (vector shapes, company badges, icons)
4. Abstract / Noise (Gaussian noise, geometric patterns, solid tints)
5. Non-CIFAR Concepts (human faces, blueprints, starfields)
"""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "ood_validation"

def get_font(size=14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", size)
        except Exception:
            return ImageFont.load_default()

def get_mono_font(size=12):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
    except Exception:
        return get_font(size)

def create_documents_and_text(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    font = get_font(12)
    mono_font = get_mono_font(11)
    bold_font = get_font(16)

    # 1. Invoice / Bill Document
    img1 = Image.new("RGB", (256, 256), color=(252, 252, 250))
    draw = ImageDraw.Draw(img1)
    draw.rectangle([16, 16, 240, 240], outline=(200, 200, 200), width=1)
    draw.text((24, 24), "INVOICE #INV-2026", fill=(30, 41, 59), font=bold_font)
    draw.line([(24, 48), (232, 48)], fill=(100, 116, 139), width=2)
    draw.text((24, 60), "Client: Enterprise Corp", fill=(51, 65, 85), font=font)
    draw.text((24, 80), "Date: 2026-09-05", fill=(51, 65, 85), font=font)
    draw.text((24, 105), "Description         Total", fill=(71, 85, 105), font=bold_font)
    draw.line([(24, 125), (232, 125)], fill=(226, 232, 240), width=1)
    draw.text((24, 135), "1. Cloud Hosting    $240.00", fill=(51, 65, 85), font=font)
    draw.text((24, 155), "2. API Analytics    $120.00", fill=(51, 65, 85), font=font)
    draw.text((24, 175), "3. Storage SLA      $45.00", fill=(51, 65, 85), font=font)
    draw.line([(24, 200), (232, 200)], fill=(100, 116, 139), width=1)
    draw.text((120, 210), "Total: $405.00", fill=(15, 23, 42), font=bold_font)
    img1.save(out_dir / "invoice_document.png")

    # 2. Source Code Screenshot
    img2 = Image.new("RGB", (256, 256), color=(30, 30, 30))
    draw = ImageDraw.Draw(img2)
    draw.rectangle([0, 0, 256, 24], fill=(45, 45, 45))
    draw.ellipse([8, 8, 16, 16], fill=(239, 68, 68))
    draw.ellipse([22, 8, 30, 16], fill=(245, 158, 11))
    draw.ellipse([36, 8, 44, 16], fill=(34, 197, 94))
    draw.text((60, 6), "main.py — editor", fill=(156, 163, 175), font=font)
    code_lines = [
        "import numpy as np",
        "import torch",
        "",
        "class DeepTransformer:",
        "    def __init__(self, d_model):",
        "        super().__init__()",
        "        self.dim = d_model",
        "        self.fc = nn.Linear(512)",
        "",
        "    def forward(self, x):",
        "        return self.fc(x)",
    ]
    y = 36
    for line in code_lines:
        draw.text((12, y), line, fill=(220, 220, 220), font=mono_font)
        y += 18
    img2.save(out_dir / "code_snippet.png")

    # 3. Dense Printed Text Page
    img3 = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img3)
    draw.text((20, 20), "Academic Abstract Section", fill=(0, 0, 0), font=bold_font)
    for i in range(12):
        y_pos = 50 + i * 16
        draw.text((20, y_pos), f"Paragraph line {i+1}: Investigating out-of-distribution detection algorithms.", fill=(40, 40, 40), font=font)
    img3.save(out_dir / "text_page.png")

    # 4. Handwritten style note / receipt
    img4 = Image.new("RGB", (256, 256), color=(255, 250, 235))
    draw = ImageDraw.Draw(img4)
    draw.text((30, 30), "Grocery Receipt", fill=(80, 70, 60), font=bold_font)
    items = ["Milk $3.99", "Bread $2.49", "Eggs $4.29", "Coffee $8.99", "Apples $3.50"]
    for idx, item in enumerate(items):
        draw.text((30, 70 + idx * 24), f"- {item}", fill=(60, 50, 40), font=font)
    img4.save(out_dir / "receipt_memo.png")


def create_ui_screenshots(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    font = get_font(12)
    bold_font = get_font(14)

    # 1. Dialog Box / Modal
    img1 = Image.new("RGB", (256, 256), color=(226, 232, 240))
    draw = ImageDraw.Draw(img1)
    draw.rectangle([28, 48, 228, 208], fill=(255, 255, 255), outline=(203, 213, 225), width=2)
    draw.rectangle([28, 48, 228, 80], fill=(241, 245, 249))
    draw.text((40, 56), "Confirm File Deletion", fill=(15, 23, 42), font=bold_font)
    draw.text((40, 100), "Are you sure you want to", fill=(51, 65, 85), font=font)
    draw.text((40, 120), "permanently delete this item?", fill=(51, 65, 85), font=font)
    draw.rectangle([40, 160, 110, 190], fill=(239, 68, 68))
    draw.text((56, 168), "Delete", fill=(255, 255, 255), font=font)
    draw.rectangle([130, 160, 200, 190], fill=(241, 245, 249), outline=(203, 213, 225))
    draw.text((144, 168), "Cancel", fill=(51, 65, 85), font=font)
    img1.save(out_dir / "dialog_modal.png")

    # 2. Web Browser Header / URL bar
    img2 = Image.new("RGB", (256, 256), color=(248, 250, 252))
    draw = ImageDraw.Draw(img2)
    draw.rectangle([0, 0, 256, 44], fill=(226, 232, 240))
    draw.rectangle([16, 8, 240, 36], fill=(255, 255, 255), outline=(203, 213, 225))
    draw.text((28, 14), "https://dashboard.cloud.service/v2", fill=(100, 116, 139), font=font)
    draw.rectangle([16, 60, 120, 140], fill=(255, 255, 255), outline=(226, 232, 240))
    draw.rectangle([136, 60, 240, 140], fill=(255, 255, 255), outline=(226, 232, 240))
    draw.text((24, 70), "CPU Usage: 42%", fill=(15, 23, 42), font=font)
    draw.text((144, 70), "Memory: 1.2 GB", fill=(15, 23, 42), font=font)
    img2.save(out_dir / "browser_dashboard.png")

    # 3. Settings Control Panel
    img3 = Image.new("RGB", (256, 256), color=(241, 245, 249))
    draw = ImageDraw.Draw(img3)
    draw.text((20, 20), "System Settings", fill=(15, 23, 42), font=bold_font)
    for i, opt in enumerate(["Dark Mode", "Auto Updates", "Notifications", "Cloud Sync"]):
        y = 60 + i * 40
        draw.rectangle([20, y, 236, y + 32], fill=(255, 255, 255), outline=(226, 232, 240))
        draw.text((32, y + 8), opt, fill=(30, 41, 59), font=font)
        draw.rectangle([200, y + 8, 224, y + 24], fill=(59, 130, 246) if i % 2 == 0 else (203, 213, 225))
    img3.save(out_dir / "settings_panel.png")


def create_logos_and_symbols(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    font = get_font(18)

    # 1. Tech Hexagonal Logo
    img1 = Image.new("RGB", (256, 256), color=(15, 23, 42))
    draw = ImageDraw.Draw(img1)
    pts = [(128, 40), (210, 85), (210, 175), (128, 220), (46, 175), (46, 85)]
    draw.polygon(pts, fill=(99, 102, 241), outline=(165, 180, 252))
    draw.polygon([(128, 70), (180, 100), (180, 160), (128, 190), (76, 160), (76, 100)], fill=(15, 23, 42))
    draw.text((80, 118), "NEXUS", fill=(255, 255, 255), font=font)
    img1.save(out_dir / "hex_tech_logo.png")

    # 2. Geometric Circular Emblem
    img2 = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img2)
    draw.ellipse([32, 32, 224, 224], outline=(225, 29, 72), width=6)
    draw.ellipse([56, 56, 200, 200], outline=(244, 63, 94), width=3)
    draw.polygon([(128, 64), (192, 176), (64, 176)], fill=(225, 29, 72))
    img2.save(out_dir / "circular_emblem.png")

    # 3. Simple Black-on-White Wordmark
    img3 = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img3)
    large_font = get_font(28)
    draw.text((44, 110), "BRANDNAME", fill=(0, 0, 0), font=large_font)
    draw.line([(44, 150), (212, 150)], fill=(0, 0, 0), width=3)
    img3.save(out_dir / "wordmark_logo.png")


def create_abstract_noise(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # 1. Pure Uniform Gaussian Noise
    noise = np.random.normal(128, 40, (256, 256, 3)).clip(0, 255).astype(np.uint8)
    Image.fromarray(noise).save(out_dir / "gaussian_noise.png")

    # 2. Solid Color Canvas (e.g. flat sky blue)
    solid = np.full((256, 256, 3), (14, 165, 233), dtype=np.uint8)
    Image.fromarray(solid).save(out_dir / "solid_flat_color.png")

    # 3. High-frequency black-and-white grid pattern
    grid = np.zeros((256, 256, 3), dtype=np.uint8)
    grid[::16, :, :] = 255
    grid[:, ::16, :] = 255
    Image.fromarray(grid).save(out_dir / "geometric_grid.png")

    # 4. Diagonal Gradient
    grad = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for x in range(256):
            grad[y, x, 0] = int((x / 256) * 255)
            grad[y, x, 1] = int((y / 256) * 255)
            grad[y, x, 2] = int(((x + y) / 512) * 255)
    Image.fromarray(grad).save(out_dir / "color_gradient.png")


def create_non_cifar_photos(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Architectural Blueprint (cyan background, white geometric lines)
    img1 = Image.new("RGB", (256, 256), color=(10, 60, 140))
    draw = ImageDraw.Draw(img1)
    for x in range(0, 256, 20):
        draw.line([(x, 0), (x, 256)], fill=(20, 80, 180), width=1)
    for y in range(0, 256, 20):
        draw.line([(0, y), (256, y)], fill=(20, 80, 180), width=1)
    draw.rectangle([40, 40, 216, 216], outline=(255, 255, 255), width=2)
    draw.rectangle([60, 60, 140, 120], outline=(255, 255, 255), width=1)
    draw.rectangle([150, 60, 200, 160], outline=(255, 255, 255), width=1)
    draw.arc([70, 130, 130, 190], 0, 180, fill=(255, 255, 255), width=2)
    img1.save(out_dir / "blueprint_floorplan.png")

    # 2. Starfield / Deep Space (black, random tiny stars, colored nebula)
    np.random.seed(123)
    space = np.zeros((256, 256, 3), dtype=np.float32)
    # Nebula glow
    for y in range(256):
        for x in range(256):
            d1 = math.hypot(x - 100, y - 120)
            d2 = math.hypot(x - 170, y - 160)
            space[y, x, 0] += max(0, 120 - d1 * 1.5)
            space[y, x, 2] += max(0, 140 - d2 * 1.2)
    # Stars
    star_x = np.random.randint(0, 256, 120)
    star_y = np.random.randint(0, 256, 120)
    star_b = np.random.uniform(150, 255, 120)
    for sx, sy, sb in zip(star_x, star_y, star_b):
        space[sy, sx] = [sb, sb, sb]
    Image.fromarray(space.clip(0, 255).astype(np.uint8)).save(out_dir / "deep_space_stars.png")

    # 3. Human Face Silhouette on sunset
    img3 = Image.new("RGB", (256, 256), color=(251, 146, 60))
    draw = ImageDraw.Draw(img3)
    draw.ellipse([40, 20, 216, 196], fill=(234, 88, 12))
    # Head and shoulders profile
    draw.ellipse([88, 50, 168, 140], fill=(24, 24, 27))
    draw.polygon([(48, 256), (88, 170), (168, 170), (208, 256)], fill=(24, 24, 27))
    img3.save(out_dir / "person_silhouette.png")

    # 4. Barcode / QR Code graphic
    img4 = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img4)
    # Barcode bars
    np.random.seed(99)
    x = 24
    while x < 232:
        w = np.random.choice([2, 4, 6, 8])
        if np.random.rand() > 0.4:
            draw.rectangle([x, 40, x + w, 180], fill=(0, 0, 0))
        x += w + np.random.choice([2, 4, 6])
    draw.text((64, 195), "9 780201 379624", fill=(0, 0, 0), font=get_font(14))
    img4.save(out_dir / "barcode_scan.png")

    # 5. Electronic Circuit Schematic
    img5 = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img5)
    draw.line([(30, 80), (80, 80)], fill=(0, 0, 0), width=2)
    # Resistor zigzag
    draw.line([(80, 80), (90, 65), (100, 95), (110, 65), (120, 95), (130, 65), (140, 80)], fill=(0, 0, 0), width=2)
    draw.line([(140, 80), (220, 80)], fill=(0, 0, 0), width=2)
    draw.line([(220, 80), (220, 180)], fill=(0, 0, 0), width=2)
    draw.line([(220, 180), (140, 180)], fill=(0, 0, 0), width=2)
    # Capacitor
    draw.line([(140, 160), (140, 200)], fill=(0, 0, 0), width=3)
    draw.line([(130, 160), (130, 200)], fill=(0, 0, 0), width=3)
    draw.line([(130, 180), (30, 180)], fill=(0, 0, 0), width=2)
    draw.line([(30, 180), (30, 80)], fill=(0, 0, 0), width=2)
    draw.text((95, 100), "R1 = 10kΩ", fill=(0, 0, 0), font=get_font(11))
    draw.text((105, 205), "C1 = 100µF", fill=(0, 0, 0), font=get_font(11))
    img5.save(out_dir / "circuit_schematic.png")


def main():
    print("Generating curated OOD validation dataset...")
    create_documents_and_text(OUT_ROOT / "documents_text")
    create_ui_screenshots(OUT_ROOT / "ui_screenshots")
    create_logos_and_symbols(OUT_ROOT / "logos_symbols")
    create_abstract_noise(OUT_ROOT / "abstract_noise")
    create_non_cifar_photos(OUT_ROOT / "non_cifar_photos")
    
    total = sum(len(files) for _, _, files in os.walk(OUT_ROOT))
    print(f"Successfully generated {total} OOD validation images in: {OUT_ROOT}")

if __name__ == "__main__":
    main()
