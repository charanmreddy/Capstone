# Multimodal Diabetic Foot Ulcer Detection and Severity Grading

A multimodal system that detects diabetic foot ulcers (DFU) and grades severity by combining RGB, thermal and plantar pressure images through late fusion. It comes with a Streamlit app that stores patient history and generates PDF reports.

## Why Multimodal

Diabetic foot ulcers are usually assessed by visual inspection, which varies between clinicians. Each modality adds different information:
- RGB images show surface appearance
- Thermal images show inflammation and tissue stress
- Plantar pressure maps show how load is spread across the foot

Combining them gives a more reliable result than any single modality.

## Repository Structure

| Folder | Contents |
|---|---|
| `Baseline Models/` | Independent models: EfficientNet-B3 (RGB), Random Forest (thermal), XGBoost (plantar) |
| `Clean-Prep-Feature/` | Cleaning, preprocessing and feature extraction notebooks for all three modalities |
| `Features_T_P/` | Extracted thermal and plantar feature sets |
| `CrossModelPairing/` | Pseudo-pairing of the three datasets (they share no patients) |
| `LateFusion/` | Late fusion of the three model outputs |
| `FusionWeightOptimization/` | Grid search, Nelder-Mead refinement and 5-fold validation of fusion weights |
| `Severity Grading/` | Severity pipeline on the STANDUP dataset: preprocessing, classifiers, fusion, weight optimization |
| `dfu_app/` | Streamlit app, inference pipeline, patient database, threshold script, test case generator |

## Binary Detection (DFU / No DFU)

| Modality | Model | Baseline Accuracy |
|---|---|---|
| RGB | EfficientNet-B3 | 97.32% |
| Thermal | Random Forest (9 features) | 96.25% |
| Plantar Pressure | XGBoost (11 features) | 68.28% |

Plantar accuracy rose from 64.65% to 90.33% after extending the features from 5 to 11. The three datasets share no patients, so records were paired by clinically informed sampling and fused with optimized weights.

**Fusion result:** weights RGB 0.898, thermal 0.055, plantar 0.046. Fused accuracy 94.31%, AUC 0.9856 (RGB alone 0.9697). The decision threshold of 0.5552 was chosen with Youden's J on the holdout set.

## Severity Grading (Healthy / R0 / R1 / R2)

| Modality | Model | Accuracy | QWK |
|---|---|---|---|
| RGB | EfficientNet-B3 with CORAL ordinal head | 50.0% | 0.5694 |
| Thermal | Random Forest cascade of 3 binary models (9 features) | 38.1% | 0.4139 |

This pipeline uses the STANDUP dataset, where RGB and thermal images come from the same patients (415 pairs), so no pseudo-pairing is needed. Fusion weights are RGB 0.60 and thermal 0.40. Final result on unseen patients: accuracy 48.8%, QWK 0.522. Grades follow the STANDUP risk scale, not the Wagner scale.

## Running the App (`dfu_app/`)

**1. Setup (Windows, from the repository root)**
```
cd dfu_app
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install scikit-learn==1.6.1 xgboost==3.4.1
```
On macOS or Linux, activate with `source venv/bin/activate`. The last line matches the library versions the models were saved with.

**2. Add the model files**

Place these five files in `dfu_app/models/` (see `models/models_instruction.txt`):
- `best_efficientnet_b3.pth`
- `best_rf_thermal.joblib` (or `best_th_thermal.joblib`)
- `plantar_xgb.json` (or `best_xgb_plantar.joblib`)
- `best_rgb_severity_coral.pth`
- `best_thermal_severity_ordinal.joblib`

**3. Check that the models load**
```
python pipeline.py
```
It should print `Models loaded OK`.

**4. Start the app**
```
streamlit run app.py
```
Open `http://localhost:8501`. Enter the patient name, age, blood glucose (mg/dL), date and up to three images. Any one or two images can be left out. The app shows the ulcer score out of 100, the label and the severity grade, and offers a PDF report. A returning patient (same name and age) gets the same ID, and the PDF compares the current visit with the previous one. Visits are stored in `dfu.db`, which is created on first run.

**5. Find the decision threshold (optional)**
```
cd thresh
python thresh.py
```
Prints the Youden threshold (0.5552) from `fusion_holdout_probs.csv`. To change the threshold, edit `BIN_THRESHOLD` in `pipeline.py`.

**6. Generate test cases (optional)**
```
cd triplet
python triplet.py
```
Builds `test_triplets/` with one folder per paired RGB, thermal and plantar case. The folder name shows the true label (`_DFU` or `_NoDFU`). Upload the three files from one folder into the app.

**Demo mode:** when all three images are uploaded, the severity grade comes from ulcer score bands set in `pipeline.py` (`DEMO_CUTOFFS`), and the app and PDF say so. Set `DEMO_SEVERITY = False` to use the severity model.

## Team

Charan M Reddy, Charan Tej A, Adarsha S R, Suhan C K
Guide: Prof. Neha Sharma
