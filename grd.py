# ============================================================
# Syriac OCR - Combined Dual Interface (Fixed Tabs)
# ============================================================

import os
import random
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
import gradio as gr
from ultralytics import YOLO

# 1) فحص الجهاز
DEVICE = 0 if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
print(f"🖥️ Running on Device: {DEVICE}")

# 2) أسماء الفئات
SYRIAC_NAMES = [
    "ܐ", "ܒ", "ܓ", "ܕ", "ܗ", "ܘ", "ܙ", "ܚ", "ܛ", "ܝ", "ܟ",
    "ܠ", "ܡ", "ܢ", "ܣ", "ܥ", "ܦ", "ܨ", "ܩ", "ܪ", "ܫ", "ܬ"
]
CLASS_COUNT = len(SYRIAC_NAMES)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ------------------------------------------------------------
# تحميل المجموعة الأولى
# ------------------------------------------------------------
MODEL_PATH_1 = Path("/content/drive/MyDrive/Syriac_OCR_Training_Results/syriac_manual100_60epochs/weights/best.pt")
DATASET_ROOT_1 = Path("/content/drive/MyDrive/Syriac_OCR_Manual100_60Epochs")

model_1 = YOLO(str(MODEL_PATH_1)) if MODEL_PATH_1.exists() else None

images_1 = []
if DATASET_ROOT_1.exists():
    for sub in ["train/images", "valid/images"]:
        p = DATASET_ROOT_1 / sub
        if p.exists():
            images_1.extend([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])

images_1 = sorted(images_1, key=lambda x: x.name)[:100]
choices_1 = [f"{i + 1:03d} - {img.name}" for i, img in enumerate(images_1)] if images_1 else ["لا توجد صور"]
map_1 = {choices_1[i]: images_1[i] for i in range(len(images_1))}

# ------------------------------------------------------------
# تحميل المجموعة الثانية
# ------------------------------------------------------------
MODEL_PATH_2 = Path("/content/drive/MyDrive/Syriac_OCR_Training_Results/syriac_datalabel_manual100_60epochs/weights/best.pt")
DATASET_ROOT_2 = Path("/content/drive/MyDrive/Syriac_OCR_DataLabel_Manual100_60Epochs")

model_2 = YOLO(str(MODEL_PATH_2)) if MODEL_PATH_2.exists() else None

images_2 = []
if DATASET_ROOT_2.exists():
    for sub in ["train/images", "valid/images"]:
        p = DATASET_ROOT_2 / sub
        if p.exists():
            images_2.extend([f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS])

images_2 = sorted(images_2, key=lambda x: x.name.lower())[:100]
choices_2 = [f"{i + 1:03d} - {img.name}" for i, img in enumerate(images_2)] if images_2 else ["لا توجد صور"]
map_2 = {choices_2[i]: images_2[i] for i in range(len(images_2))}

# ------------------------------------------------------------
# دالة المعالجة العامة
# ------------------------------------------------------------
def process_ocr(selected_name, map_dict, model_obj, dataset_root, confidence):
    if not selected_name or selected_name not in map_dict:
        blank = Image.new("RGB", (300, 300), "white")
        return blank, blank, "رجاء اختر صورة صالحة."

    image_path = map_dict[selected_name]
    gt_image = Image.open(image_path).convert("RGB")
    draw_gt = ImageDraw.Draw(gt_image)
    w, h = gt_image.size
    stem = image_path.stem

    # قراءة Ground Truth
    label_path = dataset_root / "train/labels" / f"{stem}.txt"
    if not label_path.exists():
        label_path = dataset_root / "valid/labels" / f"{stem}.txt"

    gt_boxes = []
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cls = int(float(parts[0]))
                        x, y, bw, bh = map(float, parts[1:])
                        gt_boxes.append({"cls": cls, "x": x, "y": y, "w": bw, "h": bh})
                    except Exception:
                        continue

    for box in gt_boxes:
        x1 = max(0, min(w - 1, int((box["x"] - box["w"] / 2) * w)))
        y1 = max(0, min(h - 1, int((box["y"] - box["h"] / 2) * h)))
        x2 = max(0, min(w - 1, int((box["x"] + box["w"] / 2) * w)))
        y2 = max(0, min(h - 1, int((box["y"] + box["h"] / 2) * h)))
        draw_gt.rectangle([x1, y1, x2, y2], outline="#00FF00", width=2)

    # التنبؤ
    pred_image = Image.open(image_path).convert("RGB")
    predictions = []
    
    if model_obj:
        results = model_obj.predict(source=str(image_path), conf=float(confidence), imgsz=640, verbose=False, device=DEVICE)
        draw_pred = ImageDraw.Draw(pred_image)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                predictions.append({"cls": cls, "conf": conf})
                draw_pred.rectangle([x1, y1, x2, y2], outline="#FF0000", width=2)

    report = f"### 📄 معلومات الصورة: `{image_path.name}`\n"
    report += f"- **GT Boxes:** `{len(gt_boxes)}` | **Prediction Boxes:** `{len(predictions)}`"
    return gt_image, pred_image, report

# ------------------------------------------------------------
# بناء الواجهة بصيغة Tabs
# ------------------------------------------------------------
with gr.Blocks(title="Syriac OCR - Dual Sets") as demo:
    gr.Markdown("# 🔤 Syriac OCR Dual Interface")

    with gr.Tab("المجموعة الأولى (100)"):
        with gr.Row():
            dd1 = gr.Dropdown(choices=choices_1, value=choices_1[0], label="اختر صورة")
            sl1 = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
        btn1 = gr.Button("🔍 عرض الصورة", variant="primary")
        with gr.Row():
            gt1 = gr.Image(label="🟢 Ground Truth", type="pil")
            pr1 = gr.Image(label="🔴 Prediction", type="pil")
        rep1 = gr.Markdown()
        
        btn1.click(lambda name, c: process_ocr(name, map_1, model_1, DATASET_ROOT_1, c), [dd1, sl1], [gt1, pr1, rep1])
        dd1.change(lambda name, c: process_ocr(name, map_1, model_1, DATASET_ROOT_1, c), [dd1, sl1], [gt1, pr1, rep1])

    with gr.Tab("المجموعة الثانية (100)"):
        with gr.Row():
            dd2 = gr.Dropdown(choices=choices_2, value=choices_2[0], label="اختر صورة من المجموعة الثانية")
            sl2 = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence")
        btn2 = gr.Button("🔍 عرض الصورة", variant="primary")
        with gr.Row():
            gt2 = gr.Image(label="🟢 Ground Truth", type="pil")
            pr2 = gr.Image(label="🔴 Prediction", type="pil")
        rep2 = gr.Markdown()
        
        btn2.click(lambda name, c: process_ocr(name, map_2, model_2, DATASET_ROOT_2, c), [dd2, sl2], [gt2, pr2, rep2])
        dd2.change(lambda name, c: process_ocr(name, map_2, model_2, DATASET_ROOT_2, c), [dd2, sl2], [gt2, pr2, rep2])

demo.launch(share=True, debug=False)