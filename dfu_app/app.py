import datetime as dt
import cv2, numpy as np, pandas as pd, streamlit as st
import pipeline, store

st.set_page_config(page_title="DFU Diagnostic Assistant", page_icon="9️⃣3️⃣", layout="wide")
store.init()


@st.cache_resource(show_spinner="Loading models...")
def models_ready():
    return pipeline.get()


def decode(f):
    return None if f is None else cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)


st.title("Diabetic Foot Ulcer - Multimodal Diagnostic Assistant")
models_ready()
tab_new, tab_hist = st.tabs(["New assessment", "Patient history"])

with tab_new:
    with st.form("f"):
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("Patient name")
        age = c2.number_input("Age", 0, 120, 50)
        glu = c3.number_input("Blood glucose (mg/dL)", 20.0, 800.0, 120.0, step=1.0)
        date = c4.date_input("Date", dt.date.today())
        u1, u2, u3 = st.columns(3)
        files = {"rgb": u1.file_uploader("RGB foot photo", type=["jpg", "jpeg", "png", "bmp"]),
                 "thermal": u2.file_uploader("Thermal image", type=["jpg", "jpeg", "png", "bmp"]),
                 "plantar": u3.file_uploader("Plantar pressure image", type=["jpg", "jpeg", "png", "bmp"])}
        go = st.form_submit_button("Run analysis", type="primary")
    if go:
        imgs = {k: decode(f) for k, f in files.items()}
        bad = [k for k, f in files.items() if f is not None and imgs[k] is None]
        imgs = {k: v for k, v in imgs.items() if v is not None}
        if bad:
            st.warning(f"Could not read: {', '.join(bad)} - skipped.")
        if not name.strip():
            st.error("Enter the patient name.")
        elif not imgs:
            st.error("Upload at least one image.")
        else:
            with st.spinner("Analysing..."):
                res = pipeline.predict(**imgs)
                pid = store.make_pid(name, age)
                vid = store.save_visit(pid, name.strip(), age, glu, str(date), res, {k: store.small_jpg(v) for k, v in imgs.items()})
                cur, prev, pat = store.get_visit(vid), store.prev_visit(pid, vid), store.get_patient(pid)
                st.session_state["last"] = dict(pat=pat, cur=cur, prev=prev, pdf=store.make_pdf(pat, cur, prev))
    L = st.session_state.get("last")
    if L:
        cur, prev, pat = L["cur"], L["prev"], L["pat"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Patient ID", pat["pid"])
        m2.metric("Ulcer score (/100)", cur["bin_score"], None if not prev else round(cur["bin_score"] - prev["bin_score"], 1), delta_color="inverse")
        m3.metric("Detection", cur["bin_label"])
        m4.metric("Severity", cur["sev_grade"] or "N/A")
        st.caption("Modalities used: " + ", ".join(cur["modalities"]))
        if cur["probs"].get("demo_sev"):
            st.caption("Demo mode: severity derived from ulcer score bands.")
        if prev:
            st.info(f"Returning patient - compared with visit on {prev['visit_date']} (score {prev['bin_score']}, severity {prev['sev_grade'] or 'N/A'}).")
        else:
            st.info("New patient - first visit recorded.")
        st.dataframe(pd.DataFrame([{"Modality": k.title(), "Risk probability": round(v, 3)} for k, v in cur["probs"]["bin"].items()]), hide_index=True)
        st.download_button("⬇ Download PDF report", L["pdf"], file_name=f"{pat['pid']}_{cur['visit_date']}.pdf", mime="application/pdf")

with tab_hist:
    q = st.text_input("Search by name or patient ID")
    rows = store.search(q)
    if rows:
        sel = st.selectbox("Patient", [f"{r['pid']} | {r['name']} | age {r['age']}" for r in rows])
        st.dataframe(pd.DataFrame(store.history(sel.split(" | ")[0])), hide_index=True)
    else:
        st.write("No patients yet.")