# Multimodal Diabetic Foot Ulcer Detection & Severity Grading

A multimodal system that detects diabetic foot ulcers and grades their severity (Wagner scale) by combining three imaging modalities — RGB, thermal, and plantar pressure — through a late fusion approach.

## Why Multimodal

Diabetic foot ulcers are usually assessed through manual visual inspection, which is inconsistent and depends on clinician experience. Each imaging modality captures specific information:
- RGB images show surface appearance
- Thermal images reveal inflammation and tissue stress
- Plantar pressure maps show load distribution across the foot

Combining all three gives a more reliable assessment than any single modality alone.

## Repository Structure

| Folder | Contents |
|---|---|
| `Baseline-Models/` | Independent baseline models — EfficientNet-B3 (RGB), Random Forest (thermal), XGBoost (plantar) |
| `Clean-Prep-Feature/` | Data cleaning, preprocessing, and feature extraction notebooks for all three modalities |
| `Features/` | Extracted feature sets from all three modality datasets |
| `Fusion/` | Late fusion implementation and weight optimization |
| `Improved-Plantar-Model/` | Enhanced plantar pressure model (feature/model improvements over baseline) |
| `Final-Tuned-Models/` | Final hyperparameter-tuned versions of all three modality models |
| `Cross-Modal-Validation/` | Cross-modal validation strategy and supporting code |
| `Results/` | Performance metrics and comparisons — fused system vs individual baselines |

## Modalities & Models

| Modality | Model | Baseline Accuracy |
|---|---|---|
| RGB | EfficientNet-B3 | 97.32% |
| Thermal | Random Forest | 96.25% |
| Plantar Pressure | XGBoost | 68.28% |

## Team

Charan M Reddy, Charan Tej A, Adarsha S R, Suhan C K \ 
Guide: Prof. Neha Sharma
