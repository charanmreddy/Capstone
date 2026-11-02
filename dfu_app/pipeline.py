"""Inference pipeline. Preprocessing/feature code is copied from the project notebooks."""
import os
import cv2, joblib, numpy as np, torch, torch.nn as nn
import scipy.ndimage as ndimage
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from torchvision import models, transforms

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
W_BIN = {"rgb": 0.898, "thermal": 0.055, "plantar": 0.046}
W_SEV = {"rgb": 0.60, "thermal": 0.40}

DEMO_SEVERITY = True 
DEMO_NEEDS_ALL_3 = True 
DEMO_HEALTHY = "Healthy"
DEMO_BASE = "R0" 
DEMO_CUTOFFS = [(65, "R1"), (75, "R2")]  
BIN_THRESHOLD = 0.5552   # Youden threshold from the holdout analysis
GRADES = ["Healthy", "R0 (low risk)", "R1 (medium risk)", "R2 (high risk)"]
TH_COLS = ["mean_temp", "max_temp", "std_temp", "range_temp", "hot_spot_count",
           "temp_gradient", "glcm_contrast", "glcm_homogeneity", "glcm_entropy"]
PL_COLS = ["mean_pressure", "max_pressure", "contact_area", "cop_x", "cop_y",
           "forefoot_ratio", "midfoot_ratio", "hindfoot_ratio",
           "glcm_contrast", "glcm_homogeneity", "glcm_entropy"]
_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


# ---------------- models ----------------
class CoralLayer(nn.Module):
    def __init__(self, i, n):
        super().__init__()
        self.fc = nn.Linear(i, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(n - 1))

    def forward(self, x):
        return self.fc(x) + self.bias


def _path(*names):
    for n in names:
        p = os.path.join(MODEL_DIR, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Put '{names[0]}' inside the models/ folder")


def _load():
    dev = torch.device("cpu")
    m = {}
    b = models.efficientnet_b3(weights=None)
    inf = b.classifier[1].in_features
    b.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(inf, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    b.load_state_dict(torch.load(_path("best_efficientnet_b3.pth"), map_location=dev))
    m["rgb_bin"] = b.eval()
    s = models.efficientnet_b3(weights=None)
    s.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(inf, 256), nn.ReLU(), nn.Dropout(0.3))
    s.coral = CoralLayer(256, 4)
    s.load_state_dict(torch.load(_path("best_rgb_severity_coral.pth"), map_location=dev))
    m["rgb_sev"] = s.eval()
    m["th_bin"] = joblib.load(_path("best_rf_thermal.joblib", "best_th_thermal.joblib"))
    jp = os.path.join(MODEL_DIR, "plantar_xgb.json")
    if os.path.exists(jp):                       # version-proof format
        import xgboost as xgb
        x = xgb.XGBClassifier()
        x.load_model(jp)
        m["pl_bin"] = x
    else:
        m["pl_bin"] = joblib.load(_path("best_xgb_plantar.joblib"))
    m["th_sev"] = joblib.load(_path("best_thermal_severity_ordinal.joblib"))
    return m


_M = {}


def get():
    if not _M:
        _M.update(_load())
    return _M


# ---------------- RGB ----------------
_LUT = np.array([((i / 255.0) ** (1 / 1.2)) * 255 for i in range(256)]).astype(np.uint8)


def rgb_pre(bgr, resize_first=True):
    """resize_first=True: binary pipeline (clean->224->preprocess). False: STANDUP (preprocess->224)."""
    if resize_first:
        bgr = cv2.resize(bgr, (224, 224))
    r = bgr.astype(np.float32)
    a = r.reshape(-1, 3).mean(0)
    x = np.clip(r * (a.mean() / (a + 1e-6)), 0, 255).astype(np.uint8)          # white balance
    x = cv2.LUT(x, _LUT)                                                        # gamma 1.2
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)                                    # skin ROI
    mask = cv2.inRange(hsv, np.array([0, 20, 70], np.uint8), np.array([25, 255, 255], np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k)
    if np.sum(mask) >= 0.05 * mask.size:
        x = cv2.bitwise_and(x, x, mask=mask)
    l, a_, b_ = cv2.split(cv2.cvtColor(x, cv2.COLOR_BGR2LAB))                   # CLAHE
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    x = cv2.cvtColor(cv2.merge([l, a_, b_]), cv2.COLOR_LAB2BGR)
    return x if resize_first else cv2.resize(x, (224, 224))


def _tensor(bgr):
    return _TF(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0)



# ---------------- Thermal ----------------
def _foot_mask(u8):
    _, t = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    c = cv2.morphologyEx(t, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    o = cv2.morphologyEx(c, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cs, _ = cv2.findContours(o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return o
    m = np.zeros_like(o)
    cv2.drawContours(m, [max(cs, key=cv2.contourArea)], -1, 255, -1)
    return m


def th_bin_img(gray):
    g = cv2.resize(gray, (224, 224))
    d = cv2.bilateralFilter(g, 9, 75, 75)
    cal = d.astype(np.float32) / 255.0 * 20.0 + 20.0
    u8 = ((cal - cal.min()) / (float(cal.max() - cal.min()) or 1.0) * 255).astype(np.uint8)
    roi = cal.copy()
    roi[_foot_mask(u8) == 0] = cal.min()
    r = float(roi.max() - roi.min())
    n = np.zeros_like(roi) if r == 0 else (roi - roi.min()) / r
    return (n * 255).astype(np.uint8).astype(np.float32) / 255.0   # same save/reload as training


def th_sev_img(gray):
    d = cv2.bilateralFilter(gray, 9, 75, 75)
    r = cv2.bitwise_and(d, d, mask=_foot_mask(d))
    n = (r.astype(np.float32) - r.min()) / (float(r.max()) - float(r.min()) + 1e-8)
    return cv2.resize(n, (224, 224))


def _glcm(img):
    u8 = (img * 255).astype(np.uint8)
    gl = graycomatrix(u8, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256, symmetric=True, normed=True)
    g = gl / (gl.sum() + 1e-10)
    g = g[g > 0]
    return (float(graycoprops(gl, "contrast")[0, 0]), float(graycoprops(gl, "homogeneity")[0, 0]),
            float(-np.sum(g * np.log2(g))))


def th_feats(img):
    img = img.astype(np.float32)
    hot = (img > np.percentile(img, 90)).astype(np.uint8)
    n, _ = cv2.connectedComponents(hot)
    u8 = (img * 255).astype(np.uint8)
    gx = cv2.Sobel(u8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_64F, 0, 1, ksize=3)
    c, h, e = _glcm(img)
    return {"mean_temp": float(img.mean()), "max_temp": float(img.max()), "std_temp": float(img.std()),
            "range_temp": float(img.max() - img.min()), "hot_spot_count": float(n - 1),
            "temp_gradient": float(np.mean(np.sqrt(gx ** 2 + gy ** 2))),
            "glcm_contrast": c, "glcm_homogeneity": h, "glcm_entropy": e}


# ---------------- Plantar ----------------
def _align(img):
    _, b = cv2.threshold((img * 255).astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return img
    mo = cv2.moments(max(cs, key=cv2.contourArea))
    if mo["m00"] == 0:
        return img
    ang = np.degrees(0.5 * np.arctan2(2 * mo["mu11"] / mo["m00"], mo["mu20"] / mo["m00"] - mo["mu02"] / mo["m00"]))
    if abs(ang) > 5:
        h, w = img.shape
        return cv2.warpAffine(img, cv2.getRotationMatrix2D((w // 2, h // 2), ang, 1.0), (w, h),
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img


def pl_img(gray):
    g = cv2.resize(gray, (224, 224)).astype(np.float32) / 70.0
    mn, mx = g.min(), g.max()
    n = np.zeros_like(g) if mx == mn else (g - mn) / (mx - mn)
    f = cv2.medianBlur((n * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
    s = ndimage.gaussian_filter(_align(f), sigma=2.0)
    ok, buf = cv2.imencode(".jpg", (s * 255).astype(np.uint8))                 # training saved as .jpg
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def pl_feats(img):
    h, w = img.shape
    pos = img[img > 0]
    tot = img.sum()
    if tot == 0:
        cx, cy = 0.5, 0.5
    else:
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = float((xx * img).sum() / tot / w), float((yy * img).sum() / tot / h)
    t = h // 3
    pr = [float(p[p > 0].mean()) if np.any(p > 0) else 0.0 for p in (img[:t], img[t:2 * t], img[2 * t:])]
    s = sum(pr)
    fr = [x / s for x in pr] if s else [0.33] * 3
    c, hm, e = _glcm(img)
    return {"mean_pressure": float(pos.mean()) if len(pos) else 0.0, "max_pressure": float(img.max()),
            "contact_area": float(np.sum(img > 0.05) / (h * w) * 100), "cop_x": cx, "cop_y": cy,
            "forefoot_ratio": fr[0], "midfoot_ratio": fr[1], "hindfoot_ratio": fr[2],
            "glcm_contrast": c, "glcm_homogeneity": hm, "glcm_entropy": e}


# ---------------- fusion + predict ----------------
def _fuse(vals, W):
    ks = [k for k, v in vals.items() if v is not None]
    if not ks:
        return None
    s = sum(W[k] for k in ks)
    return sum(W[k] * np.asarray(vals[k]) for k in ks) / s


def _gray(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if x.ndim == 3 else x


def _demo_grade(score, detected):
    if not detected:
        return DEMO_HEALTHY
    g = DEMO_BASE
    for c, name in sorted(DEMO_CUTOFFS):
        if score > c:
            g = name
    return g


def predict(rgb=None, thermal=None, plantar=None):
    """Each arg: BGR/gray uint8 array or None. Missing modalities are skipped (weights renormalised)."""
    m = get()
    P, F, used = {}, {}, []
    sev = {"rgb": None, "thermal": None}
    with torch.no_grad():
        if rgb is not None:
            used.append("RGB")
            P["rgb"] = float(torch.softmax(m["rgb_bin"](_tensor(rgb_pre(rgb, True))), 1)[0, 1])
            s = m["rgb_sev"]
            sev["rgb"] = torch.sigmoid(s.coral(s.classifier(s.features(_tensor(rgb_pre(rgb, False))).mean([2, 3]))))[0].numpy()
    if thermal is not None:
        used.append("Thermal")
        g = _gray(thermal)
        F["thermal"] = th_feats(th_bin_img(g))
        rf = m["th_bin"]
        cl = list(rf.classes_)
        P["thermal"] = float(rf.predict_proba(np.array([[F["thermal"][c] for c in TH_COLS]]))[0, cl.index(1) if 1 in cl else 1])
        fs = th_feats(th_sev_img(g))
        X = np.array([[fs[c] for c in TH_COLS]])
        sev["thermal"] = np.array([c.predict_proba(X)[0, 1] for c in m["th_sev"]])
    if plantar is not None:
        used.append("Plantar")
        F["plantar"] = pl_feats(pl_img(_gray(plantar)))
        pr = m["pl_bin"].predict_proba(np.array([[F["plantar"][c] for c in PL_COLS]]))[0]
        P["plantar"] = float(pr[0] + pr[2])           # flat foot + over-arch = abnormal risk
    fb = _fuse(P, W_BIN)
    fs_ = _fuse(sev, W_SEV)
    idx = int((fs_ > 0.5).sum()) if fs_ is not None else None
    grade = None if idx is None else GRADES[idx]
    score = round(float(fb) * 100, 1)
    detected = fb >= BIN_THRESHOLD
    demo = DEMO_SEVERITY and (len(used) == 3 or (not DEMO_NEEDS_ALL_3 and len(used) > 0))
    if demo:
        grade = _demo_grade(score, detected)
        idx = ([DEMO_HEALTHY, DEMO_BASE] + [n for _, n in sorted(DEMO_CUTOFFS)]).index(grade)
    return {"used": used, "features": F,
            "probs": {"bin": P, "sev_rgb": None if sev["rgb"] is None else sev["rgb"].tolist(),
                      "sev_thermal": None if sev["thermal"] is None else sev["thermal"].tolist(),
                      "sev_fused": None if fs_ is None else fs_.tolist(), "demo_sev": bool(demo)},
            "bin_score": score, "bin_label": "DFU detected" if detected else "No DFU",
            "sev_idx": idx, "sev_grade": grade}


if __name__ == "__main__":      # smoke test with random images
    r = np.random.randint(0, 255, (300, 300, 3), np.uint8)
    out = predict(rgb=r, thermal=r, plantar=r)
    print("Models loaded OK. Score:", out["bin_score"], out["bin_label"], "| Severity:", out["sev_grade"])
    print("Thermal-only :", predict(thermal=r)["bin_score"], "| Plantar-only:", predict(plantar=r)["sev_grade"])