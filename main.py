import os
import io
import time

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from inference_sdk import InferenceHTTPClient

# ─── CONFIG ────────────────────────────────────────────────────────────────────
APP_TITLE = "MedVision AI"
APP_SUBTITLE = "Intelligent Medical Scan Analysis Platform"
DEFAULT_MIN_CONFIDENCE = 0.25

# Scan type → Roboflow model ID mapping (swap with your trained model IDs)
SCAN_MODELS = {
    "Brain MRI": "brain-tumor-m2pbp-elkdc/1",
    "Chest X-Ray": "chest-xray-pneumonia-detection-pfeqf/1",
    "CT Scan (General)": "ct-scan-dgzcv-hfgd9/1",
    "Bone X-Ray": "xray-fracture-v0kdz-cj637/1",
}

FINDING_DESCRIPTIONS = {
    "tumor": "A region of abnormal tissue growth has been identified. Further histopathological analysis is recommended.",
    "pneumonia": "Consolidation or infiltrate patterns suggest possible pneumonia. Clinical correlation advised.",
    "fracture": "A discontinuity in bone architecture has been detected. Orthopedic consultation recommended.",
    "effusion": "Fluid accumulation detected in the pleural space. Monitoring and clinical evaluation advised.",
    "nodule": "A small well-defined opacity has been detected. Follow-up imaging in 3–6 months recommended.",
    "edema": "Increased fluid density patterns suggest possible edema. Cardiology or nephrology review suggested.",
    "mass": "A soft-tissue mass has been identified. Urgent specialist referral is recommended.",
    "lesion": "An area of abnormal tissue change is present. Further characterisation required.",
    "normal": "No significant abnormality detected in the scanned region.",
    "healthy": "Structures appear within normal limits.",
}

SEVERITY_MAP = {
    "tumor": "HIGH",
    "mass": "HIGH",
    "fracture": "MEDIUM",
    "pneumonia": "MEDIUM",
    "effusion": "MEDIUM",
    "nodule": "LOW",
    "edema": "MEDIUM",
    "lesion": "MEDIUM",
    "normal": "NONE",
    "healthy": "NONE",
}

SEVERITY_COLOR = {
    "HIGH": "#ef4444",
    "MEDIUM": "#f97316",
    "LOW": "#eab308",
    "NONE": "#22c55e",
}


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    if "ROBOFLOW_API_KEY" in st.secrets:
        return st.secrets["ROBOFLOW_API_KEY"]
    return os.getenv("ROBOFLOW_API_KEY", "")


def build_client(api_key: str) -> InferenceHTTPClient:
    return InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=api_key)


def read_image(image_bytes: bytes) -> np.ndarray | None:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(image)
    except Exception:
        return None


def draw_predictions(image: np.ndarray, predictions: list) -> np.ndarray:
    frame = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(frame)
    w, h = frame.size

    for pred in predictions:
        x, y = pred["x"], pred["y"]
        bw, bh = pred["width"], pred["height"]
        label = pred["class"]
        conf = pred["confidence"]

        x1 = max(0, int(x - bw / 2))
        y1 = max(0, int(y - bh / 2))
        x2 = min(w, int(x + bw / 2))
        y2 = min(h, int(y + bh / 2))

        severity = SEVERITY_MAP.get(label.lower(), "LOW")
        color_hex = SEVERITY_COLOR.get(severity, "#ffffff")
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        color_rgb = (r, g, b)

        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=2)

        label_text = f"{label.upper()}  {conf:.0%}"
        text_bbox = draw.textbbox((0, 0), label_text)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        label_top = max(0, y1 - th - 8)
        label_bottom = label_top + th + 6
        draw.rectangle([x1, label_top, x1 + tw + 8, label_bottom], fill=color_rgb)
        draw.text((x1 + 4, label_top + 3), label_text, fill=(255, 255, 255))

    return np.array(frame)


def generate_report(predictions: list, scan_type: str, patient_id: str) -> dict:
    findings = []
    max_severity_rank = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}
    overall_severity = "NONE"

    for pred in predictions:
        label = pred["class"].lower()
        conf = pred["confidence"]
        severity = SEVERITY_MAP.get(label, "LOW")
        desc = FINDING_DESCRIPTIONS.get(label, "An abnormality has been detected. Specialist review recommended.")
        findings.append({"label": label.title(), "confidence": conf, "severity": severity, "description": desc})
        if max_severity_rank.get(severity, 0) > max_severity_rank.get(overall_severity, 0):
            overall_severity = severity

    overall_status = "NORMAL" if overall_severity == "NONE" else "ABNORMAL"

    return {
        "patient_id": patient_id,
        "scan_type": scan_type,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": overall_status,
        "overall_severity": overall_severity,
        "total_findings": len(findings),
        "findings": findings,
    }


def log_session(report: dict):
    with open("medvision_log.txt", "a", encoding="utf-8") as f:
        f.write(
            f"[{report['timestamp']}] Patient: {report['patient_id']} | "
            f"Scan: {report['scan_type']} | Status: {report['overall_status']} | "
            f"Findings: {report['total_findings']}\n"
        )


def severity_badge(severity: str) -> str:
    color = SEVERITY_COLOR.get(severity, "#888")
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700;">{severity}</span>'


# ─── UI COMPONENTS ──────────────────────────────────────────────────────────────

def render_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main { background: #0b1120; }
    .stApp { background: #0b1120; }

    h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: #e2e8f0 !important; }

    .brand-header {
        display: flex; align-items: center; gap: 16px;
        padding: 28px 0 8px 0;
    }
    .brand-logo {
        width: 52px; height: 52px; border-radius: 14px;
        background: linear-gradient(135deg, #06b6d4, #3b82f6);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.6rem;
    }
    .brand-title { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: #e2e8f0; line-height: 1; }
    .brand-sub { font-size: 0.85rem; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 4px; }

    .divider { border: none; border-top: 1px solid #1e293b; margin: 20px 0; }

    .report-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
    }
    .report-card h4 { font-family: 'DM Serif Display', serif; color: #e2e8f0; margin-bottom: 4px; }

    .finding-row {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        display: flex; justify-content: space-between; align-items: flex-start;
    }

    .stat-box {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .stat-value { font-size: 2rem; font-weight: 700; color: #06b6d4; font-family: 'DM Serif Display', serif; }
    .stat-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }

    .status-normal {
        background: #052e16; border: 1px solid #16a34a; border-radius: 12px;
        padding: 16px 20px; color: #4ade80; font-weight: 600; font-size: 1.05rem;
    }
    .status-abnormal {
        background: #2d0909; border: 1px solid #ef4444; border-radius: 12px;
        padding: 16px 20px; color: #f87171; font-weight: 600; font-size: 1.05rem;
    }

    .disclaimer {
        background: #111827; border: 1px solid #1e293b; border-radius: 10px;
        padding: 14px 18px; font-size: 0.78rem; color: #475569; margin-top: 24px;
        line-height: 1.6;
    }

    .stButton>button {
        background: linear-gradient(135deg, #06b6d4, #3b82f6) !important;
        color: white !important; border: none !important; border-radius: 10px !important;
        font-weight: 600 !important; padding: 10px 28px !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: opacity 0.2s !important;
    }
    .stButton>button:hover { opacity: 0.85 !important; }

    [data-testid="stSidebar"] {
        background: #080e1a !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] * { color: #94a3b8 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #e2e8f0 !important; }

    .stSelectbox label, .stSlider label, .stRadio label,
    .stTextInput label, .stFileUploader label { color: #94a3b8 !important; }

    .stSelectbox > div > div, .stTextInput > div > input {
        background: #111827 !important; border-color: #1e293b !important;
        color: #e2e8f0 !important; border-radius: 10px !important;
    }

    .stAlert { border-radius: 10px !important; }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="brand-header">
        <div class="brand-logo">🔬</div>
        <div>
            <div class="brand-title">MedVision AI</div>
            <div class="brand-sub">Intelligent Medical Scan Analysis Platform</div>
        </div>
    </div>
    <hr class="divider">
    """, unsafe_allow_html=True)


def render_sidebar() -> tuple[str, str, float, bool]:
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Settings")
        st.markdown("---")

        scan_type = st.selectbox("Scan Type", list(SCAN_MODELS.keys()), help="Select the type of medical scan being analysed.")
        patient_id = st.text_input("Patient ID / Reference", value="PT-0001", help="Enter a patient identifier for the report.")
        min_conf = st.slider("Detection Confidence Threshold", 0.10, 0.90, DEFAULT_MIN_CONFIDENCE, 0.05,
                             help="Only show detections above this confidence level.")
        show_raw = st.checkbox("Show Raw Predictions JSON", value=False)

        st.markdown("---")
        st.markdown("### 📋 About")
        st.markdown("""
        <p style='font-size:0.8rem;color:#475569;line-height:1.6;'>
        MedVision AI uses computer vision models to assist in identifying potential anomalies in medical imaging.
        <br><br>
        <b style='color:#64748b;'>⚠️ Not a substitute for professional medical diagnosis.</b>
        </p>
        """, unsafe_allow_html=True)

    return scan_type, patient_id, min_conf, show_raw


def render_upload_zone():
    st.markdown("### 📤 Upload Scan")
    col1, col2 = st.columns([3, 1])
    with col1:
        source = st.radio("Input Source", ["Upload File", "Use Camera"], horizontal=True)
    if source == "Upload File":
        return st.file_uploader("Drop a scan image here (JPG / PNG / DICOM preview)", type=["jpg", "jpeg", "png"])
    else:
        return st.camera_input("Capture scan via camera")


def render_report(report: dict):
    status = report["overall_status"]
    severity = report["overall_severity"]
    sev_color = SEVERITY_COLOR.get(severity, "#888")

    st.markdown("---")
    st.markdown("### 📄 Analysis Report")

    # Header card
    st.markdown(f"""
    <div class="report-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <h4 style='margin:0;'>Scan Report — {report['scan_type']}</h4>
                <p style='color:#64748b;font-size:0.82rem;margin:4px 0 0;'>Patient: <b style='color:#94a3b8;'>{report['patient_id']}</b>  ·  Generated: {report['timestamp']}</p>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.3rem;font-weight:700;color:{sev_color};">{status}</div>
                <div style="font-size:0.75rem;color:#475569;">Overall Severity: {severity}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{report["total_findings"]}</div><div class="stat-label">Findings</div></div>', unsafe_allow_html=True)
    with c2:
        high_count = sum(1 for f in report["findings"] if f["severity"] == "HIGH")
        st.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ef4444;">{high_count}</div><div class="stat-label">High Severity</div></div>', unsafe_allow_html=True)
    with c3:
        avg_conf = np.mean([f["confidence"] for f in report["findings"]]) if report["findings"] else 0
        st.markdown(f'<div class="stat-box"><div class="stat-value">{avg_conf:.0%}</div><div class="stat-label">Avg Confidence</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Overall status banner
    if status == "NORMAL":
        st.markdown('<div class="status-normal">✅ No significant abnormalities detected. Structures appear within normal limits.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-abnormal">⚠️ Abnormal findings detected. Immediate clinical review is advised.</div>', unsafe_allow_html=True)

    # Findings detail
    if report["findings"]:
        st.markdown("<br>**Detailed Findings**", unsafe_allow_html=True)
        for f in report["findings"]:
            badge = severity_badge(f["severity"])
            st.markdown(f"""
            <div class="finding-row">
                <div>
                    <div style="font-weight:600;color:#e2e8f0;margin-bottom:4px;">{f['label']}</div>
                    <div style="font-size:0.82rem;color:#64748b;max-width:520px;">{f['description']}</div>
                </div>
                <div style="text-align:right;min-width:110px;">
                    {badge}
                    <div style="font-size:0.78rem;color:#64748b;margin-top:6px;">Confidence: {f['confidence']:.0%}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <b>⚠️ Clinical Disclaimer:</b> MedVision AI is an assistive tool intended to support — not replace — professional medical judgement.
        All findings must be reviewed and validated by a qualified radiologist or clinician before any clinical decisions are made.
        This tool is not FDA-approved for diagnostic use.
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="MedVision AI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_css()
    render_header()

    api_key = get_api_key()
    if not api_key:
        st.error("🔑 **API key not found.** Set `ROBOFLOW_API_KEY` in Streamlit secrets or as an environment variable.")
        st.stop()

    scan_type, patient_id, min_conf, show_raw = render_sidebar()
    uploaded = render_upload_zone()

    if uploaded is None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("📂 Upload a medical scan image using the controls above to begin analysis.")
        return

    image_bytes = uploaded.getvalue()
    image = read_image(image_bytes)
    if image is None:
        st.error("Could not decode the uploaded image. Please try a different file.")
        return

    model_id = SCAN_MODELS.get(scan_type, list(SCAN_MODELS.values())[0])
    client = build_client(api_key)

    with st.spinner("🔍 Analysing scan — please wait…"):
        try:
            result = client.infer(image, model_id=model_id)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return

    raw_predictions = result.get("predictions", []) if isinstance(result, dict) else []
    predictions = [p for p in raw_predictions if float(p.get("confidence", 0)) >= min_conf]

    annotated = draw_predictions(image, predictions)
    report = generate_report(predictions, scan_type, patient_id)
    log_session(report)

    # Display annotated image
    st.markdown("### 🖼️ Annotated Scan")
    col_img, col_orig = st.columns(2)
    with col_img:
        st.image(annotated,
                 caption=f"AI Analysis — {scan_type}", use_container_width=True)
    with col_orig:
        st.image(image,
                 caption="Original Scan", use_container_width=True)

    render_report(report)

    if show_raw:
        with st.expander("🔧 Raw Predictions JSON"):
            st.json(raw_predictions)

    if report["findings"]:
        st.download_button(
            label="⬇️ Download Report (JSON)",
            data=str(report).replace("'", '"'),
            file_name=f"medvision_report_{patient_id}_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()

