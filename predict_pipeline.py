# -*- coding: utf-8 -*-
"""predict_pipeline.py
خط أنابيب الاستدلال والتشخيص الميداني المتكامل (End-to-End Inference Pipeline)
يقوم بتحميل النموذج المحفوظ best_convnext.pth وتطبيق المعالجة الحرارية الثلاثية وتشخيص أي لوح شمسي جديد.
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as T
import timm

# 12 Classes and Arabic descriptions
CLASS_TRANSLATIONS = {
    'Cell': 'بقعة ساخنة في خلية مفردة (Single Hot Cell)',
    'Cell-Multi': 'بقع ساخنة في عدة خلايا (Multiple Hot Cells)',
    'Cracking': 'تشقق أو كسر ميكانيكي بالسطح (Cell Cracking)',
    'Diode': 'عطل أو تفعيل دايود الالتفاف (Bypass Diode Fault - 1/3)',
    'Diode-Multi': 'عطل دايودات الالتفاف المتعددة (Multi Diodes Fault - 2/3)',
    'Hot-Spot': 'نقطة ساخنة في لوح رقيق (Hot Spot - Thin Film)',
    'Hot-Spot-Multi': 'نقاط ساخنة متعددة (Multi Hot Spots - Thin Film)',
    'No-Anomaly': 'لوح سليم وطبيعي (Nominal / No-Anomaly)',
    'Offline-Module': 'لوح مفصول أو معطل بالكامل (Offline Module)',
    'Shadowing': 'حجب وظلال جزئية (Shadowing Obstruction)',
    'Soiling': 'تراكم غبار وتلوث سطحي (Soiling Accumulation)',
    'Vegetation': 'حجب بأعشاب ونباتات (Vegetation Overgrowth)'
}

IMG_H, IMG_W = 160, 96
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

eval_transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=MEAN, std=STD),
])

def preprocess_thermal_3ch(img_gray: np.ndarray, target_h=IMG_H, target_w=IMG_W):
    """تحويل الصورة الحرارية إلى 3 قنوات متخصصة: الخام + CLAHE + Fixed-Scale Sobel."""
    resized = cv2.resize(img_gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4))
    ch_clahe = clahe.apply(resized)
    gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad_fixed = np.clip(grad / 160.0 * 255.0, 0, 255).astype(np.uint8)
    return resized, ch_clahe, grad_fixed, np.stack([resized, ch_clahe, grad_fixed], axis=-1)


def load_model(ckpt_path="best_convnext.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"لم يتم العثور على ملف النموذج: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("arch", "convnext_tiny.fb_in22k_ft_in1k")
    class_names = ckpt.get("class_names", list(CLASS_TRANSLATIONS.keys()))

    model = timm.create_model(arch, pretrained=False, num_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, class_names, device


def inspect_solar_image(image_path, ckpt_path="best_convnext.pth", output_fig="inspection_report.png"):
    model, class_names, device = load_model(ckpt_path)

    raw_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw_gray is None:
        raise ValueError(f"تعذر فتح الصورة: {image_path}")

    # 1. تطبيق خط المعالجة الحرارية الثلاثي
    ch_raw, ch_clahe, ch_sobel, img_3ch = preprocess_thermal_3ch(raw_gray)
    pil_img = Image.fromarray(img_3ch)
    input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)

    # 2. الاستدلال بواسطة 4x TTA لضمان أعلى ثقة
    with torch.no_grad():
        p1 = F.softmax(model(input_tensor), dim=1)
        p2 = F.softmax(model(torch.flip(input_tensor, dims=[3])), dim=1)
        p3 = F.softmax(model(torch.flip(input_tensor, dims=[2])), dim=1)
        p4 = F.softmax(model(torch.flip(input_tensor, dims=[2, 3])), dim=1)
        probs = ((p1 + p2 + p3 + p4) / 4.0).squeeze().cpu().numpy()

    top_idx = int(np.argmax(probs))
    top_class = class_names[top_idx]
    top_prob = float(probs[top_idx] * 100.0)

    # أفضل 3 احتمالات
    top3_idx = np.argsort(probs)[-3:][::-1]

    # 3. رسم تقرير الفحص البصري للمهندس / الفني
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5))
    
    axes[0].imshow(ch_raw, cmap='inferno')
    axes[0].set_title("1. Raw Thermal IR", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(ch_clahe, cmap='magma')
    axes[1].set_title("2. CLAHE Thermal Contrast", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(ch_sobel, cmap='gray')
    axes[2].set_title("3. Sobel Edge / Gradients", fontsize=11, fontweight='bold')
    axes[2].axis('off')

    # عرض القنوات الثلاث مدمجة
    axes[3].imshow(img_3ch)
    axes[3].set_title(f"4. Composite 3-Channel Input\nResult: {top_class} ({top_prob:.1f}%)", fontsize=11, fontweight='bold', color='navy')
    axes[3].axis('off')

    ar_desc = CLASS_TRANSLATIONS.get(top_class, '')
    plt.suptitle(f"SOLAR PV INSPECTION: {top_class} [{ar_desc}] — Confidence: {top_prob:.1f}%", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*70)
    print(" تقرير الفحص الفني للوح الشمسي (SOLAR PV DIAGNOSTIC REPORT)")
    print("="*70)
    print(f"مسار الصورة:        {image_path}")
    print(f"التشخيص الأساسي:    {top_class}")
    print(f"التوصيف بالعربية:   {ar_desc}")
    print(f"نسبة الثقة (TTA):   {top_prob:.2f}%")
    print("\nأعلى 3 احتمالات تشخيصية:")
    for rank, idx in enumerate(top3_idx, 1):
        cname = class_names[idx]
        print(f"  {rank}. {cname:18s} ({probs[idx]*100:6.2f}%) -> {CLASS_TRANSLATIONS.get(cname, '')}")
    print(f"\nتم حفظ لوحة الفحص البصري في: {output_fig}")
    print("="*70 + "\n")

    return {
        "class": top_class,
        "arabic": ar_desc,
        "confidence": top_prob,
        "top3": [{"class": class_names[i], "prob": float(probs[i]*100)} for i in top3_idx],
        "report_image": output_fig
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Solar PV Module Inference Pipeline")
    parser.add_argument('--image', type=str, required=True, help="Path to infrared solar image")
    parser.add_argument('--ckpt', type=str, default="best_convnext.pth", help="Checkpoint path")
    parser.add_argument('--output', type=str, default="inspection_report.png", help="Output report image")
    args = parser.parse_args()

    inspect_solar_image(args.image, ckpt_path=args.ckpt, output_fig=args.output)

