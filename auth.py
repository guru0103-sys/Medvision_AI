import time, hashlib, json, os
import streamlit as st
from datetime import datetime

USERS = {
    "admin":       {"password": hashlib.sha256("admin123".encode()).hexdigest(),    "role": "Admin",       "name": "Administrator"},
    "doctor":      {"password": hashlib.sha256("doctor123".encode()).hexdigest(),   "role": "Doctor",      "name": "Dr. Smith"},
    "radiologist": {"password": hashlib.sha256("radio123".encode()).hexdigest(),    "role": "Radiologist", "name": "Dr. Patel"},
}

ROLE_PERMISSIONS = {
    "Admin":       {"can_analyse":True,"can_compare":True,"can_export_pdf":True,"can_view_logs":True,  "color":"#ef4444"},
    "Doctor":      {"can_analyse":True,"can_compare":True,"can_export_pdf":True,"can_view_logs":False, "color":"#22c55e"},
    "Radiologist": {"can_analyse":True,"can_compare":True,"can_export_pdf":True,"can_view_logs":True,  "color":"#3b82f6"},
}

SESSION_TIMEOUT = 900
AUDIT_LOG_FILE  = "audit_log.json"

def write_audit(username, role, action, detail=""):
    entry = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "username": username, "role": role, "action": action, "detail": detail}
    logs = []
    if os.path.exists(AUDIT_LOG_FILE):
        try:
            with open(AUDIT_LOG_FILE) as f: logs = json.load(f)
        except: pass
    logs.append(entry)
    with open(AUDIT_LOG_FILE, "w") as f: json.dump(logs[-500:], f, indent=2)

def read_audit_log():
    if not os.path.exists(AUDIT_LOG_FILE): return []
    try:
        with open(AUDIT_LOG_FILE) as f: return json.load(f)
    except: return []

def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()

def verify_login(username, password):
    user = USERS.get(username.lower())
    return user and user["password"] == hash_password(password)

def init_session():
    for k,v in [("logged_in",False),("username",""),("role",""),("name",""),("login_time",None),("last_active",None)]:
        if k not in st.session_state: st.session_state[k] = v

def check_timeout():
    if not st.session_state.get("logged_in"): return False
    last = st.session_state.get("last_active")
    return last and (time.time() - last) > SESSION_TIMEOUT

def refresh_activity(): st.session_state["last_active"] = time.time()

def logout():
    if st.session_state.get("logged_in"):
        write_audit(st.session_state["username"], st.session_state["role"], "LOGOUT")
    for k in ["logged_in","username","role","name","login_time","last_active"]:
        st.session_state[k] = False if k == "logged_in" else None

def get_permissions():
    return ROLE_PERMISSIONS.get(st.session_state.get("role","Doctor"), ROLE_PERMISSIONS["Doctor"])

def render_login():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
    .stApp,.main{background:#060d18;}
    .stTextInput>div>input{background:#111827 !important;border-color:#1e293b !important;color:#e2e8f0 !important;border-radius:10px !important;}
    .stButton>button{background:linear-gradient(135deg,#06b6d4,#3b82f6) !important;color:white !important;border:none !important;border-radius:10px !important;font-weight:600 !important;width:100% !important;}
    </style>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("""
        <div style="text-align:center;padding:48px 0 32px;">
            <div style="width:68px;height:68px;border-radius:18px;background:linear-gradient(135deg,#06b6d4,#3b82f6);
                display:inline-flex;align-items:center;justify-content:center;font-size:2.2rem;
                margin-bottom:20px;box-shadow:0 0 32px rgba(6,182,212,0.4);">🔬</div>
            <div style="font-family:'DM Serif Display',serif;font-size:2.2rem;color:#e2e8f0;">MedVision AI</div>
            <div style="font-size:0.75rem;color:#475569;letter-spacing:0.12em;text-transform:uppercase;margin-top:6px;">
                Secure Clinical Access Portal
            </div>
        </div>
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:16px;padding:32px 28px;margin-bottom:16px;">
        """, unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        login_clicked = st.button("Sign In →", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0a1525;border:1px solid #1e293b;border-radius:10px;padding:14px 18px;font-size:0.74rem;color:#475569;line-height:2;">
            <b style="color:#64748b;">Demo credentials</b><br>
            Admin &nbsp;&nbsp;&nbsp;&nbsp;→ admin / admin123<br>
            Doctor &nbsp;&nbsp;&nbsp;→ doctor / doctor123<br>
            Radiologist → radiologist / radio123
        </div>""", unsafe_allow_html=True)

    if login_clicked:
        if verify_login(username, password):
            user = USERS[username.lower()]
            st.session_state.update({"logged_in":True,"username":username.lower(),"role":user["role"],"name":user["name"],"login_time":time.time(),"last_active":time.time()})
            write_audit(username.lower(), user["role"], "LOGIN", "Successful login")
            st.rerun()
        else:
            st.error("Invalid username or password.")
            write_audit(username.lower(), "Unknown", "LOGIN_FAILED", f"Failed attempt")
