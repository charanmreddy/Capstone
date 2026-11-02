"""SQLite storage, patient ID, PDF report."""
import datetime as dt, hashlib, io, json, os, sqlite3
import cv2
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether

DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dfu.db")
KEYS = ["rgb", "thermal", "plantar"]
NAMES = {"rgb": "RGB", "thermal": "Thermal", "plantar": "Plantar"}


def _c():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _c() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS patients(pid TEXT PRIMARY KEY, name TEXT, age INTEGER, created TEXT);
        CREATE TABLE IF NOT EXISTS visits(id INTEGER PRIMARY KEY AUTOINCREMENT, pid TEXT, visit_date TEXT, saved_at TEXT,
          modalities TEXT, bin_score REAL, bin_label TEXT, sev_grade TEXT, sev_idx INTEGER, probs TEXT, features TEXT,
          rgb_img BLOB, thermal_img BLOB, plantar_img BLOB, glucose REAL);""")
        if "glucose" not in [r[1] for r in c.execute("PRAGMA table_info(visits)")]:
            c.execute("ALTER TABLE visits ADD COLUMN glucose REAL")


def make_pid(name, age):
    key = f"{' '.join(name.lower().split())}|{int(age)}"
    return "DFU-" + hashlib.sha1(key.encode()).hexdigest()[:8].upper()


def small_jpg(img, side=600):
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def save_visit(pid, name, age, glucose, date, res, imgs):
    now = dt.datetime.now().isoformat(timespec="seconds")
    with _c() as c:
        c.execute("INSERT OR IGNORE INTO patients(pid,name,age,created) VALUES(?,?,?,?)", (pid, name, int(age), now))
        cur = c.execute(
            "INSERT INTO visits(pid,visit_date,saved_at,modalities,bin_score,bin_label,sev_grade,sev_idx,probs,features,"
            "glucose,rgb_img,thermal_img,plantar_img) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, date, now, json.dumps(res["used"]), res["bin_score"], res["bin_label"], res["sev_grade"],
             res["sev_idx"], json.dumps(res["probs"]), json.dumps(res["features"]), float(glucose),
             imgs.get("rgb"), imgs.get("thermal"), imgs.get("plantar")))
        return cur.lastrowid


def _visit(r):
    if r is None:
        return None
    d = dict(r)
    d["modalities"], d["probs"], d["features"] = json.loads(d["modalities"]), json.loads(d["probs"]), json.loads(d["features"])
    d["imgs"] = {k: d[k + "_img"] for k in KEYS if d[k + "_img"]}
    return d


def get_visit(vid):
    with _c() as c:
        return _visit(c.execute("SELECT * FROM visits WHERE id=?", (vid,)).fetchone())


def prev_visit(pid, vid):
    with _c() as c:
        return _visit(c.execute("SELECT * FROM visits WHERE pid=? AND id<? ORDER BY id DESC LIMIT 1", (pid, vid)).fetchone())


def get_patient(pid):
    with _c() as c:
        return dict(c.execute("SELECT * FROM patients WHERE pid=?", (pid,)).fetchone())


def search(q=""):
    with _c() as c:
        return [dict(r) for r in c.execute("SELECT * FROM patients WHERE pid LIKE ? OR name LIKE ? ORDER BY created DESC",
                                           (f"%{q}%", f"%{q}%"))]


def history(pid):
    with _c() as c:
        return [dict(r) for r in c.execute(
            "SELECT id AS visit, visit_date, modalities, bin_score, bin_label, sev_grade, glucose AS glucose_mgdl FROM visits WHERE pid=? ORDER BY id", (pid,))]


# ---------------- PDF ----------------
def _f(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else ("-" if v is None else str(v))


def _d(a, b):
    return f"{a - b:+.3f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"


def _g(v):
    return "-" if v is None else f"{v:g}"


def _im(b, mw=7.5 * cm, mh=6 * cm):
    w, h = PILImage.open(io.BytesIO(b)).size
    s = min(mw / w, mh / h)
    return RLImage(io.BytesIO(b), width=w * s, height=h * s)


def make_pdf(pat, cur, prev=None):
    ss = getSampleStyleSheet()
    nm = ParagraphStyle("nm", parent=ss["Normal"], fontSize=8.5, leading=10)
    cap = ParagraphStyle("cap", parent=nm, fontName="Helvetica-Bold", alignment=1)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3)
    ttl = ParagraphStyle("ttl", parent=ss["Title"], fontSize=16, spaceAfter=6)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.3 * cm, rightMargin=1.3 * cm, topMargin=1.1 * cm, bottomMargin=1.1 * cm)
    W = 21 * cm - 2.6 * cm

    def tbl(rows, widths=None, head=True, fs=8.5):
        t = Table(rows, colWidths=widths, repeatRows=1 if head else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), fs), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
        if head:
            st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)]
        t.setStyle(TableStyle(st))
        return t

    E = [Paragraph("Diabetic Foot Ulcer - Multimodal Assessment Report", ttl),
         tbl([["Patient ID", pat["pid"], "Visit date", cur["visit_date"]],
              ["Name", pat["name"], "Age", pat["age"]],
              ["Blood glucose (mg/dL)", _g(cur.get("glucose")), "Modalities used", ", ".join(cur["modalities"])]],
             head=False, widths=[4 * cm, 5.2 * cm, 4 * cm, 5.2 * cm]),
         Paragraph("Results", h2)]
    sev = lambda v: v["sev_grade"] or "N/A (needs RGB or Thermal)"
    if prev:
        rows = [["Measure", "Current", f"Previous ({prev['visit_date']})", "Change"],
                ["Ulcer detection score (/100)", _f(cur["bin_score"]), _f(prev["bin_score"]), _d(cur["bin_score"], prev["bin_score"])],
                ["Detection result", cur["bin_label"], prev["bin_label"], "-"],
                ["Severity grade", sev(cur), sev(prev), "-"],
                ["Blood glucose (mg/dL)", _g(cur.get("glucose")), _g(prev.get("glucose")),
                 f"{cur['glucose'] - prev['glucose']:+g}" if cur.get("glucose") is not None and prev.get("glucose") is not None else "-"]]
        for k in KEYS:
            a, b = cur["probs"]["bin"].get(k), prev["probs"]["bin"].get(k)
            if a is not None or b is not None:
                rows.append([f"{NAMES[k]} model risk probability", _f(a), _f(b), _d(a, b)])
        E.append(tbl(rows, widths=[6.8 * cm, 3.8 * cm, 4.8 * cm, 3 * cm]))
    else:
        rows = [["Measure", "Value"], ["Ulcer detection score (/100)", _f(cur["bin_score"])],
                ["Detection result", cur["bin_label"]], ["Severity grade", sev(cur)]]
        rows += [[f"{NAMES[k]} model risk probability", _f(v)] for k, v in cur["probs"]["bin"].items()]
        E += [tbl(rows), Paragraph("First recorded visit - no previous data to compare.", ss["Italic"])]
    mods = [k for k in KEYS if cur["imgs"].get(k) or (prev or {}).get("imgs", {}).get(k)]
    if mods:
        cw = W / len(mods)

        def cell(src, label, k):
            b = src["imgs"].get(k) if src else None
            return [Paragraph(f"{NAMES[k]} - {label}", cap), _im(b, cw - 0.6 * cm, 3.5 * cm) if b else Paragraph("not provided", nm)]

        grid = [[cell(cur, "current", k) for k in mods]] + ([[cell(prev, "previous", k) for k in mods]] if prev else [])
        t = Table(grid, colWidths=[cw] * len(mods))
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        E += [Paragraph("Images", h2), t]
    blocks = []
    for k in KEYS:
        fc = cur["features"].get(k)
        if not fc:
            continue
        fp = (prev or {}).get("features", {}).get(k, {})
        rows = [["Feature", "Current"] + (["Previous", "Change"] if prev else [])]
        for n, v in fc.items():
            rows.append([n, _f(v)] + ([_f(fp.get(n)), _d(v, fp.get(n))] if prev else []))
        cws = [3.3, 1.9, 1.9, 1.9] if prev else [4.5, 2.5]
        blocks.append([Paragraph(f"<b>{NAMES[k]}</b>", nm), tbl(rows, [c * cm for c in cws], fs=7.8)])
    if blocks:
        o = Table([blocks], colWidths=[W / len(blocks)] * len(blocks))
        o.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        E.append(KeepTogether([Paragraph("Extracted features", h2), o]))
    E += [Spacer(1, 8), Paragraph("Research prototype (capstone project). Output is decision support only and not a clinical diagnosis. "
                                  "Severity uses the STANDUP risk scale (Healthy/R0/R1/R2)."
                                  + (" Demo mode: severity was derived from ulcer score bands, not from the severity model." if cur["probs"].get("demo_sev") else ""),
                                  ss["Italic"])]
    doc.build(E)
    return buf.getvalue()