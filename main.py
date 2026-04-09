import os, io, time, json, base64
import urllib.error
import urllib.parse
import urllib.request
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

from auth import (
    init_session, check_timeout, refresh_activity, logout,
    render_login, get_permissions, write_audit, read_audit_log,
    ROLE_PERMISSIONS,
)
from pdf_report import generate_pdf

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DEFAULT_MIN_CONFIDENCE = 0.25
SESSION_TIMEOUT = 900

SCAN_MODELS = {
    "Brain MRI":         "brain-tumor-m2pbp-elkdc/1",
    "Chest X-Ray":       "chest-xray-pneumonia-detection-pfeqf/1",
    "CT Scan (General)": "ct-scan-dgzcv-hfgd9/1",
    "Bone X-Ray":        "xray-fracture-v0kdz-cj637/1",
    "Retinal Scan":      "retinal-disease-detection/1",
}

FINDING_DESCRIPTIONS = {
    "tumor":     "A region of abnormal tissue growth identified. Histopathological analysis recommended.",
    "pneumonia": "Consolidation/infiltrate patterns suggest pneumonia. Clinical correlation advised.",
    "fracture":  "Discontinuity in bone architecture detected. Orthopedic consultation recommended.",
    "effusion":  "Fluid accumulation in pleural space. Monitoring and clinical evaluation advised.",
    "nodule":    "Small well-defined opacity detected. Follow-up imaging in 3-6 months recommended.",
    "edema":     "Increased fluid density patterns. Cardiology or nephrology review suggested.",
    "mass":      "Soft-tissue mass identified. Urgent specialist referral is recommended.",
    "lesion":    "Area of abnormal tissue change. Further characterisation required.",
    "normal":    "No significant abnormality detected in the scanned region.",
    "healthy":   "Structures appear within normal limits.",
}

SEVERITY_MAP = {
    "tumor":"HIGH","mass":"HIGH",
    "fracture":"MEDIUM","pneumonia":"MEDIUM","effusion":"MEDIUM","edema":"MEDIUM","lesion":"MEDIUM",
    "nodule":"LOW","normal":"NONE","healthy":"NONE",
}

SEVERITY_COLOR = {"HIGH":"#ef4444","MEDIUM":"#f97316","LOW":"#eab308","NONE":"#22c55e"}
SEVERITY_RGB   = {"HIGH":(239,68,68),"MEDIUM":(249,115,22),"LOW":(234,179,8),"NONE":(34,197,94)}


# ─── HELPERS ───────────────────────────────────────────────────────────────────

def get_api_key():
    if "ROBOFLOW_API_KEY" in st.secrets: return st.secrets["ROBOFLOW_API_KEY"]
    return os.getenv("ROBOFLOW_API_KEY","")

def infer_image(image, model_id, api_key):
    payload = io.BytesIO()
    image.save(payload, format="PNG")

    url = (
        f"https://detect.roboflow.com/{urllib.parse.quote(model_id, safe='/')}?"
        f"{urllib.parse.urlencode({'api_key': api_key})}"
    )
    request = urllib.request.Request(
        url,
        data=base64.b64encode(payload.getvalue()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return json.loads(response.read().decode(charset))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Roboflow request failed ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Roboflow request failed: {exc.reason}") from exc

def read_image(f):
    try:
        f.seek(0)
        return Image.open(f).convert("RGB")
    except Exception:
        return None

def draw_predictions(image, predictions):
    frame = image.copy(); draw = ImageDraw.Draw(frame); w,h = frame.size
    try:    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",13)
    except: font = ImageFont.load_default()
    for pred in predictions:
        x,y,bw,bh = pred["x"],pred["y"],pred["width"],pred["height"]
        label,conf = pred["class"],pred["confidence"]
        x1=max(0,int(x-bw/2)); y1=max(0,int(y-bh/2))
        x2=min(w,int(x+bw/2)); y2=min(h,int(y+bh/2))
        color = SEVERITY_RGB.get(SEVERITY_MAP.get(label.lower(),"LOW"),(255,255,255))
        for t in range(3): draw.rectangle([x1-t,y1-t,x2+t,y2+t],outline=color)
        txt = f" {label.upper()}  {conf:.0%} "
        bb  = draw.textbbox((0,0),txt,font=font)
        tw,th = bb[2]-bb[0],bb[3]-bb[1]
        ty1 = max(0,y1-th-6)
        draw.rectangle([x1,ty1,x1+tw+4,y1],fill=color)
        draw.text((x1+2,ty1+1),txt,fill=(255,255,255),font=font)
    return frame

def generate_report(predictions, scan_type, patient_id):
    findings=[]; rank={"NONE":0,"LOW":1,"MEDIUM":2,"HIGH":3}; overall="NONE"
    for pred in predictions:
        lb=pred["class"].lower(); sev=SEVERITY_MAP.get(lb,"LOW")
        findings.append({"label":lb.title(),"confidence":pred["confidence"],"severity":sev,
                         "description":FINDING_DESCRIPTIONS.get(lb,"Abnormality detected. Specialist review recommended.")})
        if rank.get(sev,0)>rank.get(overall,0): overall=sev
    return {"patient_id":patient_id,"scan_type":scan_type,"timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status":"NORMAL" if overall=="NONE" else "ABNORMAL","overall_severity":overall,
            "total_findings":len(findings),"findings":findings}

def log_session(report, username, role):
    with open("medvision_log.txt","a",encoding="utf-8") as f:
        f.write(f"[{report['timestamp']}] {username}({role}) | Patient:{report['patient_id']} | {report['scan_type']} | {report['overall_status']}\n")
    write_audit(username,role,"SCAN_ANALYSED",f"Patient:{report['patient_id']} | {report['scan_type']} | {report['overall_status']}")

def severity_badge(s):
    c=SEVERITY_COLOR.get(s,"#888")
    return f'<span style="background:{c};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:700;">{s}</span>'


# ─── ANIMATED LOADER ───────────────────────────────────────────────────────────

def show_loader():
    st.markdown("""
    <div id="mvloader" style="position:fixed;inset:0;z-index:9999;background:rgba(4,8,15,.95);
        display:flex;flex-direction:column;align-items:center;justify-content:center;backdrop-filter:blur(10px);">
        <div style="position:relative;width:130px;height:130px;margin-bottom:28px;">
            <svg width="130" height="130" style="position:absolute;animation:spinA 2s linear infinite;">
                <circle cx="65" cy="65" r="58" fill="none" stroke="rgba(0,229,255,.1)" stroke-width="3"/>
                <circle cx="65" cy="65" r="58" fill="none" stroke="#00e5ff" stroke-width="3"
                    stroke-dasharray="90 275" stroke-linecap="round"/>
            </svg>
            <svg width="130" height="130" style="position:absolute;animation:spinA 1.3s linear infinite reverse;">
                <circle cx="65" cy="65" r="42" fill="none" stroke="rgba(59,130,246,.15)" stroke-width="2"/>
                <circle cx="65" cy="65" r="42" fill="none" stroke="#3b82f6" stroke-width="2"
                    stroke-dasharray="50 215" stroke-linecap="round"/>
            </svg>
            <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
                font-size:2.4rem;animation:pulseIcon 1.4s ease-in-out infinite;">🔬</div>
        </div>
        <div style="font-family:sans-serif;font-size:1.1rem;font-weight:600;color:#e2e8f0;
            margin-bottom:8px;animation:floatTxt 1.5s ease-in-out infinite;">
            Analysing Scan...
        </div>
        <div style="font-size:0.75rem;color:#475569;letter-spacing:.1em;text-transform:uppercase;">
            AI inference in progress
        </div>
        <div style="width:220px;height:3px;background:rgba(255,255,255,.05);border-radius:3px;
            margin-top:28px;overflow:hidden;">
            <div style="height:100%;background:linear-gradient(90deg,#06b6d4,#3b82f6);
                border-radius:3px;animation:loadBar 2s ease-in-out infinite;"></div>
        </div>
    </div>
    <style>
    @keyframes spinA{from{transform:rotate(0)}to{transform:rotate(360deg)}}
    @keyframes pulseIcon{0%,100%{transform:translate(-50%,-50%) scale(1)}50%{transform:translate(-50%,-50%) scale(1.18)}}
    @keyframes floatTxt{0%,100%{opacity:.7;transform:translateY(0)}50%{opacity:1;transform:translateY(-4px)}}
    @keyframes loadBar{0%{width:0%;margin-left:0}50%{width:55%;margin-left:22%}100%{width:0%;margin-left:100%}}
    </style>""", unsafe_allow_html=True)


# ─── CONFIDENCE GAUGE ──────────────────────────────────────────────────────────

def render_gauge(avg_conf, severity):
    color = SEVERITY_COLOR.get(severity,"#06b6d4")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(avg_conf*100,1),
        number={"suffix":"%","font":{"size":28,"color":"#e2e8f0"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#334155","tickfont":{"color":"#64748b","size":9}},
            "bar":{"color":color,"thickness":0.25},
            "bgcolor":"#0f172a","bordercolor":"#1e293b",
            "steps":[{"range":[0,40],"color":"rgba(239,68,68,0.08)"},
                     {"range":[40,70],"color":"rgba(249,115,22,0.08)"},
                     {"range":[70,100],"color":"rgba(34,197,94,0.08)"}],
            "threshold":{"line":{"color":color,"width":3},"thickness":0.75,"value":avg_conf*100},
        },
        title={"text":"Detection Confidence","font":{"color":"#64748b","size":11}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font={"color":"#e2e8f0"},height=200,margin=dict(t=36,b=4,l=16,r=16))
    st.plotly_chart(fig, use_container_width=True)


# ─── CSS ───────────────────────────────────────────────────────────────────────

def render_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
    .main,.stApp{background:#060d18;}
    h1,h2,h3{font-family:'DM Serif Display',serif !important;color:#e2e8f0 !important;}
    .brand-header{display:flex;align-items:center;gap:16px;padding:20px 0 8px;}
    .brand-logo{width:48px;height:48px;border-radius:14px;background:linear-gradient(135deg,#06b6d4,#3b82f6);
        display:flex;align-items:center;justify-content:center;font-size:1.5rem;box-shadow:0 0 20px rgba(6,182,212,.3);}
    .brand-title{font-family:'DM Serif Display',serif;font-size:2rem;color:#e2e8f0;line-height:1;}
    .brand-sub{font-size:.78rem;color:#64748b;letter-spacing:.08em;text-transform:uppercase;margin-top:4px;}
    .divider{border:none;border-top:1px solid #1e293b;margin:16px 0;}
    .report-card{background:#111827;border:1px solid #1e293b;border-radius:16px;padding:24px;margin-bottom:16px;}
    .finding-row{background:#0f172a;border:1px solid #1e293b;border-radius:10px;padding:14px 18px;
        margin-bottom:10px;display:flex;justify-content:space-between;align-items:flex-start;}
    .stat-box{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:16px 20px;text-align:center;}
    .stat-value{font-size:1.9rem;font-weight:700;color:#06b6d4;font-family:'DM Serif Display',serif;}
    .stat-label{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;}
    .status-normal{background:#052e16;border:1px solid #16a34a;border-radius:12px;padding:16px 20px;color:#4ade80;font-weight:600;}
    .status-abnormal{background:#2d0909;border:1px solid #ef4444;border-radius:12px;padding:16px 20px;color:#f87171;font-weight:600;}
    .disclaimer{background:#111827;border:1px solid #1e293b;border-radius:10px;padding:14px 18px;
        font-size:.78rem;color:#475569;margin-top:24px;line-height:1.6;}
    .audit-row{background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:.76rem;}
    .stButton>button{background:linear-gradient(135deg,#06b6d4,#3b82f6) !important;
        color:white !important;border:none !important;border-radius:10px !important;font-weight:600 !important;padding:10px 28px !important;}
    .stButton>button:hover{opacity:.85 !important;}
    [data-testid="stSidebar"]{background:#060d18 !important;border-right:1px solid #1e293b !important;}
    [data-testid="stSidebar"] *{color:#94a3b8 !important;}
    [data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#e2e8f0 !important;}
    .stSelectbox label,.stSlider label,.stRadio label,.stTextInput label,.stFileUploader label{color:#94a3b8 !important;}
    .stSelectbox>div>div,.stTextInput>div>input{background:#111827 !important;border-color:#1e293b !important;color:#e2e8f0 !important;border-radius:10px !important;}
    .stTabs [data-baseweb="tab-list"]{background:#0d1929;border-radius:10px;padding:4px;}
    .stTabs [data-baseweb="tab"]{color:#64748b !important;border-radius:8px;}
    .stTabs [aria-selected="true"]{background:#1e293b !important;color:#e2e8f0 !important;}
    </style>""", unsafe_allow_html=True)


# ─── SIDEBAR ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        role  = st.session_state.get("role","")
        name  = st.session_state.get("name","")
        rinfo = ROLE_PERMISSIONS.get(role,{})
        st.markdown(f"""
        <div style="background:#0d1929;border:1px solid #1e293b;border-radius:12px;padding:16px;margin-bottom:12px;">
            <div style="font-size:.68rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;">Signed in as</div>
            <div style="font-weight:600;color:#e2e8f0;font-size:.95rem;">{name}</div>
            <div style="margin-top:6px;">
                <span style="background:{rinfo.get('color','#888')};color:#fff;
                    padding:2px 10px;border-radius:999px;font-size:.68rem;font-weight:700;">{role}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        last = st.session_state.get("last_active", time.time())
        remaining = max(0, int(SESSION_TIMEOUT-(time.time()-last)))
        mins, secs = divmod(remaining,60)
        tc = "#ef4444" if remaining<120 else "#64748b"
        st.markdown(f'<div style="font-size:.7rem;color:{tc};text-align:center;margin-bottom:12px;font-family:monospace;">⏱ Session: {mins:02d}:{secs:02d} remaining</div>', unsafe_allow_html=True)

        st.markdown("### ⚙️ Settings")
        st.markdown("---")
        scan_type  = st.selectbox("Scan Type", list(SCAN_MODELS.keys()))
        patient_id = st.text_input("Patient ID", value="PT-0001")
        min_conf   = st.slider("Confidence Threshold", 0.10, 0.90, DEFAULT_MIN_CONFIDENCE, 0.05)
        show_raw   = st.checkbox("Show Raw JSON", value=False)
        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            logout(); st.rerun()

    return scan_type, patient_id, min_conf, show_raw


# ─── HEADER ────────────────────────────────────────────────────────────────────

def render_header():
    role  = st.session_state.get("role","")
    rinfo = ROLE_PERMISSIONS.get(role,{})
    st.markdown(f"""
    <div class="brand-header">
        <div class="brand-logo">🔬</div>
        <div style="flex:1;">
            <div class="brand-title">MedVision AI</div>
            <div class="brand-sub">Intelligent Medical Scan Analysis Platform</div>
        </div>
        <span style="background:{rinfo.get('color','#888')};color:#fff;padding:5px 16px;border-radius:999px;font-size:.75rem;font-weight:700;">{role}</span>
    </div>
    <hr class="divider">""", unsafe_allow_html=True)


# ─── RENDER REPORT ─────────────────────────────────────────────────────────────

def render_report(report):
    status    = report["overall_status"]
    severity  = report["overall_severity"]
    sev_color = SEVERITY_COLOR.get(severity,"#888")

    st.markdown("---")
    st.markdown("### 📄 Analysis Report")
    st.markdown(f"""
    <div class="report-card">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-family:'DM Serif Display',serif;font-size:1.2rem;color:#e2e8f0;">
                    Scan Report — {report['scan_type']}
                </div>
                <div style="color:#64748b;font-size:.82rem;margin-top:4px;">
                    Patient: <b style="color:#94a3b8;">{report['patient_id']}</b> &nbsp;·&nbsp; {report['timestamp']}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.2rem;font-weight:700;color:{sev_color};">{status}</div>
                <div style="font-size:.72rem;color:#475569;">Severity: {severity}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    high_count = sum(f["severity"] == "HIGH" for f in report["findings"])
    avg_conf   = float(np.mean([f["confidence"] for f in report["findings"]])) if report["findings"] else 0.0

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="stat-box"><div class="stat-value">{report["total_findings"]}</div><div class="stat-label">Findings</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-box"><div class="stat-value" style="color:#ef4444;">{high_count}</div><div class="stat-label">High Severity</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="stat-box"><div class="stat-value">{avg_conf:.0%}</div><div class="stat-label">Avg Confidence</div></div>', unsafe_allow_html=True)
    with c4: render_gauge(avg_conf, severity)

    st.markdown("<br>", unsafe_allow_html=True)
    if status=="NORMAL":
        st.markdown('<div class="status-normal">✅ No significant abnormalities detected. Structures appear within normal limits.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-abnormal">⚠️ Abnormal findings detected. Immediate clinical review is advised.</div>', unsafe_allow_html=True)

    if report["findings"]:
        st.markdown("<br>**Detailed Findings**", unsafe_allow_html=True)
        for f in report["findings"]:
            st.markdown(f"""
            <div class="finding-row">
                <div>
                    <div style="font-weight:600;color:#e2e8f0;margin-bottom:4px;">{f['label']}</div>
                    <div style="font-size:.82rem;color:#64748b;max-width:520px;">{f['description']}</div>
                </div>
                <div style="text-align:right;min-width:110px;">
                    {severity_badge(f['severity'])}
                    <div style="font-size:.78rem;color:#64748b;margin-top:6px;">Confidence: {f['confidence']:.0%}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <b>⚠️ Clinical Disclaimer:</b> MedVision AI is an assistive tool. All findings must be reviewed by a
        qualified radiologist or clinician. Not FDA-approved for diagnostic use.
    </div>""", unsafe_allow_html=True)


# ─── AUDIT LOG ─────────────────────────────────────────────────────────────────

def render_audit_log():
    st.markdown("### 🔒 Audit Trail")
    logs = read_audit_log()
    if not logs:
        st.info("No audit entries yet.")
        return

    search = st.text_input("Search", placeholder="username, action, patient ID...")
    colors = {"LOGIN":"#22c55e","LOGOUT":"#64748b","LOGIN_FAILED":"#ef4444",
              "SCAN_ANALYSED":"#06b6d4","PDF_EXPORTED":"#a855f7","SCAN_COMPARED":"#f59e0b"}

    for entry in reversed(logs[-100:]):
        if search and search.lower() not in json.dumps(entry).lower():
            continue
        c = colors.get(entry.get("action",""),"#64748b")
        st.markdown(f"""
        <div class="audit-row">
            <span style="color:#475569;">{entry.get('timestamp','')}</span> &nbsp;·&nbsp;
            <b style="color:#94a3b8;">{entry.get('username','')}</b> &nbsp;·&nbsp;
            <span style="color:#64748b;">{entry.get('role','')}</span> &nbsp;·&nbsp;
            <span style="color:{c};font-weight:700;">{entry.get('action','')}</span> &nbsp;·&nbsp;
            <span style="color:#475569;">{entry.get('detail','')}</span>
        </div>""", unsafe_allow_html=True)

    st.download_button("⬇️ Export Audit Log (JSON)", data=json.dumps(logs,indent=2),
                       file_name=f"audit_{time.strftime('%Y%m%d')}.json", mime="application/json")


# ─── SINGLE ANALYSIS ───────────────────────────────────────────────────────────

def tab_analyse(scan_type, patient_id, min_conf, show_raw):
    api_key = get_api_key()
    if not api_key:
        st.error("API key not found. Set ROBOFLOW_API_KEY."); return

    perm = get_permissions()
    st.markdown("### 📤 Upload Scan")
    c1,_ = st.columns([3,1])
    with c1:
        src = st.radio("Input Source",["Upload File","Use Camera"],horizontal=True,key="src1")
    uploaded = (st.file_uploader("Drop scan image (JPG/PNG)",type=["jpg","jpeg","png"],key="up1")
                if src=="Upload File" else st.camera_input("Capture scan",key="cam1"))

    if uploaded is None:
        st.info("Upload a scan image to begin analysis."); return

    image = read_image(uploaded)
    if image is None:
        st.error("Could not decode image."); return

    model_id = SCAN_MODELS.get(scan_type, list(SCAN_MODELS.values())[0])
    ph       = st.empty()

    with ph: show_loader()
    try:
        result = infer_image(image, model_id, api_key)
    except Exception as e:
        ph.empty(); st.error(f"Inference failed: {e}"); return
    ph.empty()

    raw  = result.get("predictions",[]) if isinstance(result,dict) else []
    pred = [p for p in raw if float(p.get("confidence",0))>=min_conf]
    ann  = draw_predictions(image, pred)
    rep  = generate_report(pred, scan_type, patient_id)
    log_session(rep, st.session_state["username"], st.session_state["role"])

    st.markdown("### 🖼️ Annotated Scan")
    ca,cb = st.columns(2)
    with ca: st.image(ann,   caption=f"AI Analysis — {scan_type}", use_container_width=True)
    with cb: st.image(image, caption="Original Scan",              use_container_width=True)

    render_report(rep)

    if show_raw:
        with st.expander("Raw Predictions JSON"): st.json(raw)

    # Downloads
    dl1,dl2 = st.columns(2)
    with dl1:
        st.download_button("⬇️ Download JSON Report",
            data=json.dumps(rep,indent=2),
            file_name=f"report_{patient_id}_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json", key="json_dl")

    with dl2:
        if perm.get("can_export_pdf") and st.button("📄 Generate PDF Report", key="pdf_btn"):
            with st.spinner("Building PDF..."):
                pdf_bytes = generate_pdf(rep, ann, image,
                                         st.session_state["name"],
                                         st.session_state["role"])
            write_audit(st.session_state["username"], st.session_state["role"],
                        "PDF_EXPORTED", f"Patient: {patient_id}")
            st.download_button("⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"medvision_{patient_id}_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf", key="pdf_dl")


# ─── COMPARISON TAB ────────────────────────────────────────────────────────────

def tab_compare(scan_type, patient_id, min_conf):
    api_key = get_api_key()
    if not api_key:
        st.error("API key not found."); return

    st.markdown("### 🔄 Comparison Mode")
    st.info("Upload two scans to compare findings side by side — useful for before/after or different modalities.")

    ca,cb = st.columns(2)
    with ca:
        st.markdown("**Scan A**")
        fa = st.file_uploader("Upload Scan A", type=["jpg","jpeg","png"], key="upa")
        la = st.text_input("Label A", value="Before", key="la")
    with cb:
        st.markdown("**Scan B**")
        fb = st.file_uploader("Upload Scan B", type=["jpg","jpeg","png"], key="upb")
        lb = st.text_input("Label B", value="After",  key="lb")

    if fa is None or fb is None:
        st.info("Upload both scans to begin comparison."); return

    img_a,img_b = read_image(fa),read_image(fb)
    if img_a is None or img_b is None:
        st.error("Could not decode one or both images."); return

    model_id = SCAN_MODELS.get(scan_type, list(SCAN_MODELS.values())[0])
    ph       = st.empty()

    with ph: show_loader()
    try:
        res_a = infer_image(img_a, model_id, api_key)
        res_b = infer_image(img_b, model_id, api_key)
    except Exception as e:
        ph.empty(); st.error(f"Inference failed: {e}"); return
    ph.empty()

    def proc(res, img, pid):
        raw  = res.get("predictions",[]) if isinstance(res,dict) else []
        pred = [p for p in raw if float(p.get("confidence",0))>=min_conf]
        return draw_predictions(img, pred), generate_report(pred, scan_type, pid)

    ann_a,rep_a = proc(res_a, img_a, f"{patient_id}-A")
    ann_b,rep_b = proc(res_b, img_b, f"{patient_id}-B")

    st.markdown("### 🖼️ Side-by-Side")
    c1,c2 = st.columns(2)
    with c1: st.image(ann_a, caption=f"AI: {la}", use_container_width=True)
    with c2: st.image(ann_b, caption=f"AI: {lb}", use_container_width=True)

    st.markdown("### 📊 Comparison Summary")
    sc1,sc2 = st.columns(2)

    def summary(rep, label):
        sc = SEVERITY_COLOR.get(rep["overall_severity"],"#888")
        avg_c = float(np.mean([f["confidence"] for f in rep["findings"]])) if rep["findings"] else 0
        st.markdown(f"""
        <div class="report-card">
            <div style="font-weight:700;font-size:1rem;color:#e2e8f0;margin-bottom:8px;">{label}</div>
            <div style="color:{sc};font-weight:700;font-size:1.05rem;">{rep['overall_status']}</div>
            <div style="font-size:.78rem;color:#64748b;margin-top:4px;">
                Severity: {rep['overall_severity']} &nbsp;·&nbsp;
                Findings: {rep['total_findings']} &nbsp;·&nbsp;
                Confidence: {avg_c:.0%}
            </div>
        </div>""", unsafe_allow_html=True)
        for f in rep["findings"]:
            st.markdown(f'<div style="font-size:.78rem;color:#94a3b8;padding:3px 0 3px 8px;">• {f["label"]} — {severity_badge(f["severity"])} &nbsp; {f["confidence"]:.0%}</div>', unsafe_allow_html=True)

    with sc1: summary(rep_a, la)
    with sc2: summary(rep_b, lb)

    write_audit(st.session_state["username"], st.session_state["role"],
                "SCAN_COMPARED", f"Patient:{patient_id} | {la} vs {lb}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="MedVision AI", page_icon="🔬",
                       layout="wide", initial_sidebar_state="expanded")

    init_session()

    if check_timeout():
        st.warning("⏱ Session timed out due to inactivity. Please sign in again.")
        logout(); st.rerun()

    if not st.session_state.get("logged_in"):
        render_login(); return

    refresh_activity()
    render_css()
    render_header()

    scan_type, patient_id, min_conf, show_raw = render_sidebar()
    perm = get_permissions()

    tab_labels = ["🔬 Analyse"]
    if perm.get("can_compare"):   tab_labels.append("🔄 Compare")
    if perm.get("can_view_logs"): tab_labels.append("🔒 Audit Trail")

    tabs = st.tabs(tab_labels)

    with tabs[0]:
        tab_analyse(scan_type, patient_id, min_conf, show_raw)

    if perm.get("can_compare") and len(tabs) >= 2:
        with tabs[1]:
            tab_compare(scan_type, patient_id, min_conf)

    if perm.get("can_view_logs"):
        with tabs[-1]:
            render_audit_log()


if __name__ == "__main__":
    main()
