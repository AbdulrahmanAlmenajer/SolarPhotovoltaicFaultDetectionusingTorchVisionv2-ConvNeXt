# -*- coding: utf-8 -*-
"""evaluate_convnext.py
Evaluation of trained ConvNeXt-Tiny checkpoint on the independent test set (2,000 images) with 4x TTA.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import timm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score

from InfraredSolarModules_TorchVisionV2 import (
    CLASS_NAMES, NUM_CLASSES, test_loader, test_df, DEVICE, USE_AMP
)

@torch.no_grad()
def evaluate_checkpoint(ckpt_path="best_convnext.pth", output_cm="confusion_matrix_convnext.png"):
    print(f"Loading checkpoint: {ckpt_path} on {DEVICE}...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    arch = ckpt.get("arch", "convnext_tiny.fb_in22k_ft_in1k")
    val_f1 = ckpt.get("val_macro_f1", 0.0)
    print(f"Model Arch: {arch} | Saved Val Macro-F1: {val_f1:.4f}")

    model = timm.create_model(arch, pretrained=False, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Running evaluation with 4x Test-Time Augmentation (TTA) on {len(test_loader.dataset)} test images...")
    all_probs = []
    all_targets = []

    for imgs, targets in test_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            p1 = F.softmax(model(imgs), dim=1)
            p2 = F.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
            p3 = F.softmax(model(torch.flip(imgs, dims=[2])), dim=1)
            p4 = F.softmax(model(torch.flip(imgs, dims=[2, 3])), dim=1)
            probs = (p1 + p2 + p3 + p4) / 4.0
        all_probs.append(probs.cpu().numpy())
        all_targets.extend(targets.numpy())

    probs = np.vstack(all_probs)
    targets = np.array(all_targets)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)

    print("\n" + "="*65)
    print(" CONVNEXT-TINY TEST SET EVALUATION RESULTS ")
    print("="*65)
    print(f"TEST ACCURACY:            {acc*100:.2f}%")
    print(f"BALANCED ACCURACY:        {bal_acc*100:.2f}%")
    print(f"MACRO F1-SCORE:           {macro_f1*100:.2f}%")
    print(f"WEIGHTED F1-SCORE:        {weighted_f1*100:.2f}%")
    print("="*65 + "\n")

    report_str = classification_report(targets, preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    print(report_str)

    report_dict = classification_report(targets, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    with open("convnext_test_results.json", "w") as f:
        json.dump({
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "report": report_dict
        }, f, indent=2)

    # Plot Confusion Matrix
    cm = confusion_matrix(targets, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title("Normalized Confusion Matrix — ConvNeXt-Tiny (4x TTA)", fontweight="bold", fontsize=13)
    ax.set_ylabel("True Class", fontsize=11)
    ax.set_xlabel("Predicted Class", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_cm, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to: {output_cm}")

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix_path": output_cm
    }

if __name__ == "__main__":
    evaluate_checkpoint()

