"""
URL-Segment-Aware Phishing Detector — Streamlit Interface (COMPLETE FINAL)
==========================================================================
Run:  streamlit run app.py
"""

import re, pickle, os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tldextract
from urllib.parse import urlparse

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Animated cyber background ── */
@keyframes float1 { 0%,100%{transform:translateY(0) rotate(0deg);opacity:0.07} 50%{transform:translateY(-30px) rotate(10deg);opacity:0.13} }
@keyframes float2 { 0%,100%{transform:translateY(0) rotate(0deg);opacity:0.05} 50%{transform:translateY(-20px) rotate(-8deg);opacity:0.10} }
@keyframes float3 { 0%,100%{transform:translateY(0) rotate(0deg);opacity:0.06} 50%{transform:translateY(-25px) rotate(5deg);opacity:0.11} }
@keyframes scanline { 0%{top:-100%} 100%{top:200%} }
@keyframes pulse-glow { 0%,100%{box-shadow:0 0 30px rgba(124,58,237,0.15)} 50%{box-shadow:0 0 60px rgba(168,85,247,0.25)} }

[data-testid="stAppViewContainer"] {
    background:
        linear-gradient(135deg, #080618 0%, #0f0a2e 40%, #130d35 70%, #0a0820 100%) !important;
    min-height: 100vh;
    position: relative;
    overflow: hidden;
}

/* Cyber grid overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124,58,237,0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,58,237,0.06) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* Moving scan line */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    left: 0;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg,
        transparent,
        rgba(168,85,247,0.15),
        rgba(124,58,237,0.3),
        rgba(168,85,247,0.15),
        transparent);
    animation: scanline 8s linear infinite;
    pointer-events: none;
    z-index: 1;
}

/* Floating background icons */
.bg-icons {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.bg-icon {
    position: absolute;
    font-size: 5rem;
    filter: blur(1px);
    user-select: none;
}
.bg-icon:nth-child(1)  { top:5%;  left:3%;  font-size:4rem;  animation: float1 7s ease-in-out infinite; }
.bg-icon:nth-child(2)  { top:12%; left:88%; font-size:5rem;  animation: float2 9s ease-in-out infinite; }
.bg-icon:nth-child(3)  { top:28%; left:6%;  font-size:3rem;  animation: float3 6s ease-in-out infinite; }
.bg-icon:nth-child(4)  { top:35%; left:92%; font-size:3.5rem;animation: float1 8s ease-in-out infinite 1s; }
.bg-icon:nth-child(5)  { top:55%; left:2%;  font-size:4.5rem;animation: float2 7s ease-in-out infinite 2s; }
.bg-icon:nth-child(6)  { top:60%; left:85%; font-size:3rem;  animation: float3 9s ease-in-out infinite 1s; }
.bg-icon:nth-child(7)  { top:75%; left:10%; font-size:3.5rem;animation: float1 6s ease-in-out infinite 3s; }
.bg-icon:nth-child(8)  { top:80%; left:78%; font-size:4rem;  animation: float2 8s ease-in-out infinite 2s; }
.bg-icon:nth-child(9)  { top:90%; left:45%; font-size:3rem;  animation: float3 7s ease-in-out infinite 1s; }
.bg-icon:nth-child(10) { top:20%; left:48%; font-size:2.5rem;animation: float1 10s ease-in-out infinite 4s; }
.bg-icon:nth-child(11) { top:68%; left:55%; font-size:3rem;  animation: float2 9s ease-in-out infinite 3s; }
.bg-icon:nth-child(12) { top:42%; left:30%; font-size:2rem;  animation: float3 8s ease-in-out infinite 5s; }

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none; }

/* Make main content sit above background */
[data-testid="stMainBlockContainer"] {
    position: relative;
    z-index: 2;
}

.main-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 900;
    color: #ffffff;
    text-shadow: 0 0 30px #a855f7, 0 0 60px #7c3aed;
    margin-bottom: 0.2rem;
    padding-top: 2rem;
}
.main-subtitle {
    text-align: center;
    color: #c084fc;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Glass card effect for main content area */
[data-testid="stTextInput"] > div > div > input {
    background: rgba(30,27,75,0.85) !important;
    color: #e2e8f0 !important;
    border: 2px solid #7c3aed !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    padding: 14px 16px !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #a855f7 !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.35) !important;
}
[data-testid="stTextInput"] label {
    color: #c4b5fd !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

[data-testid="stButton"] > button {
    background: linear-gradient(90deg, #7c3aed, #a855f7) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.65rem 2.5rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(90deg, #6d28d9, #9333ea) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.6) !important;
    transform: translateY(-2px) !important;
}

.result-phishing {
    background: linear-gradient(135deg, rgba(69,10,10,0.95), rgba(127,29,29,0.95));
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1.2rem 0;
    box-shadow: 0 0 30px rgba(239,68,68,0.3);
    backdrop-filter: blur(10px);
}
.result-legit {
    background: linear-gradient(135deg, rgba(5,46,22,0.95), rgba(20,83,45,0.95));
    border: 2px solid #22c55e;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1.2rem 0;
    box-shadow: 0 0 30px rgba(34,197,94,0.3);
    backdrop-filter: blur(10px);
}
.result-title {
    font-size: 1.8rem;
    font-weight: 900;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
}
.result-stats {
    color: #d1d5db;
    font-size: 0.95rem;
    margin: 0;
}

.seg-card {
    background: rgba(30,27,75,0.85);
    border: 1px solid #4c1d95;
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin: 0.35rem 0;
    font-family: monospace;
    font-size: 0.85rem;
    color: #e2e8f0;
    backdrop-filter: blur(8px);
}
.seg-label { color: #a78bfa; font-weight: 700; }

.flag-box {
    background: rgba(28,25,23,0.9);
    border: 1px solid #78350f;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-top: 0.3rem;
    font-size: 0.88rem;
    color: #fde68a;
    line-height: 1.9;
    backdrop-filter: blur(8px);
}
.flag-safe {
    background: rgba(5,46,22,0.9);
    border: 1px solid #166534;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-top: 0.3rem;
    font-size: 0.88rem;
    color: #86efac;
    backdrop-filter: blur(8px);
}

hr { border-color: #4c1d95 !important; }

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.78rem;
    margin-top: 2.5rem;
    padding-bottom: 1.5rem;
}

#MainMenu, footer, header { visibility: hidden; }
</style>

<!-- Floating background phishing/security themed icons -->
<div class="bg-icons">
    <div class="bg-icon">🛡️</div>
    <div class="bg-icon">🔒</div>
    <div class="bg-icon">⚠️</div>
    <div class="bg-icon">🔐</div>
    <div class="bg-icon">🕵️</div>
    <div class="bg-icon">🦠</div>
    <div class="bg-icon">🔒</div>
    <div class="bg-icon">🛡️</div>
    <div class="bg-icon">💀</div>
    <div class="bg-icon">⚠️</div>
    <div class="bg-icon">🔗</div>
    <div class="bg-icon">🔐</div>
</div>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_PATH = "models/best_model.h5"
TOK_PATH   = "models/tokenizers.pkl"
SEG_PATH   = "models/seg_max.pkl"
SEGMENTS   = ["subdomain", "domain_tld", "path", "query"]

# ── URL Parser ─────────────────────────────────────────────────────────────
def parse_segments(url: str) -> dict:
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    ext    = tldextract.extract(url)
    parsed = urlparse(url)
    subdomain  = ext.subdomain.lower().strip()
    domain_tld = (ext.domain + "." + ext.suffix).lower().strip() if ext.suffix else ext.domain.lower()
    path       = parsed.path.lstrip("/").lower().strip()
    query      = parsed.query.lower().strip()
    return {
        "subdomain"  : subdomain  or "none",
        "domain_tld" : domain_tld or "none",
        "path"       : path       or "none",
        "query"      : query      or "none",
    }

# ── Load artifacts ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔮 Loading model…")
def load_artifacts():
    required = [MODEL_PATH, TOK_PATH, SEG_PATH]
    if not all(os.path.exists(p) for p in required):
        return None, None, None
    model      = tf.keras.models.load_model(MODEL_PATH)
    tokenizers = pickle.load(open(TOK_PATH, "rb"))
    seg_max    = pickle.load(open(SEG_PATH, "rb"))
    return model, tokenizers, seg_max

# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess(url: str, tokenizers: dict, seg_max: dict):
    segs = parse_segments(url)
    X = {}
    for seg in SEGMENTS:
        seq = tokenizers[seg].texts_to_sequences([segs[seg]])
        X[f"input_{seg}"] = pad_sequences(
            seq, maxlen=seg_max[seg], padding="post", truncating="post"
        )
    return X, segs

# ── Inference ──────────────────────────────────────────────────────────────
def predict(url: str, model, tokenizers: dict, seg_max: dict):
    X, segs = preprocess(url, tokenizers, seg_max)
    prob  = float(model.predict(X, verbose=0)[0][0])
    label = "Phishing" if prob >= 0.5 else "Legitimate"
    conf  = prob if prob >= 0.5 else 1.0 - prob
    return {
        "label": label, "phishing_prob": prob,
        "confidence": conf, "segments": segs
    }

# ── Risk flags ─────────────────────────────────────────────────────────────
def risk_flags(url: str, segs: dict) -> list:
    flags = []
    if re.search(r'\d+\.\d+\.\d+\.\d+', url):
        flags.append("⚠️ This link uses a number address instead of a real website name")
    if len(url) > 200:
        flags.append("⚠️ This link is unusually long — scammers often do this to hide what's inside")
    if url.count('.') > 5:
        flags.append("⚠️ Too many dots in the link — this is a common trick to confuse you")
    if '@' in url:
        flags.append("⚠️ There's an @ symbol in the link — this can be used to trick you into going somewhere else")
    if url.count('//') > 1:
        flags.append("⚠️ The link has a hidden redirect inside it")
    if re.search(r'(%[0-9a-fA-F]{2}){3,}', url):
        flags.append("⚠️ Parts of this link are hidden using codes — a common trick by scammers")
    susp_tlds = ('.xyz','.tk','.ml','.ga','.cf','.gq','.top','.click','.loan')
    if any(segs["domain_tld"].endswith(t) for t in susp_tlds):
        flags.append(f"⚠️ The website ending (.{segs['domain_tld'].split('.')[-1]}) is commonly used by fake sites")
    brands = ['paypal','amazon','google','apple','microsoft','netflix',
              'facebook','instagram','bank','secure','login','verify','update']
    if any(b in segs["subdomain"] for b in brands):
        flags.append("⚠️ A well-known brand name appears in an unusual part of the link — scammers do this to look real")
    if not url.startswith("https://"):
        flags.append("ℹ️ This link doesn't use a secure connection (no https)")
    return flags

# ── Simple Human Explanation ───────────────────────────────────────────────
def build_explanation(label, ph, segs, flags):
    domain = segs["domain_tld"]

    if label == "Phishing":
        if any(b in segs["subdomain"] for b in
               ['paypal','amazon','google','apple','microsoft',
                'netflix','facebook','instagram','bank','secure','login','verify','update']):
            reason = (
                f"It's pretending to be a well-known website by putting a famous name "
                f"at the start of the link, but the actual website is <b>{domain}</b> — "
                f"which is not the real one."
            )
        elif re.search(r'\d+\.\d+\.\d+\.\d+', segs["domain_tld"] + segs["subdomain"]):
            reason = (
                "Instead of having a proper website name, this link uses a string of numbers. "
                "Real websites don't look like this — it's a strong sign of a scam."
            )
        elif any(segs["domain_tld"].endswith(t) for t in
                 ('.xyz','.tk','.ml','.ga','.cf','.gq','.top','.click','.loan')):
            reason = (
                f"The website address ends in <b>.{domain.split('.')[-1]}</b>, "
                f"which is very commonly used by fake or scam websites."
            )
        elif len(flags) >= 3:
            reason = (
                "There are several suspicious things about this link when looked at closely — "
                "the kind of tricks scammers use to make fake links look real."
            )
        else:
            reason = (
                f"The website <b>{domain}</b> looks like it was made to trick people. "
                f"Our AI spotted patterns in this link that are very common in scam websites."
            )
        message = (
            f"⚠️ <b>Our AI thinks this link is not safe.</b><br><br>"
            f"{reason}<br><br>"
            f"<b>What you should do:</b> Don't click on it, don't enter your password or "
            f"any personal details, and don't make any payments through this link. "
            f"If someone sent you this, they may be trying to steal your information."
        )
        box_style = (
            "background:rgba(69,10,10,0.92);border:2px solid #ef4444;border-radius:14px;"
            "padding:1.4rem 1.7rem;color:#fca5a5;margin-top:1.2rem;line-height:2;"
            "font-size:0.95rem;backdrop-filter:blur(10px);"
        )
    else:
        if not flags:
            reason = (
                f"The website <b>{domain}</b> looks normal and nothing suspicious "
                f"was found in the link. It seems to be a genuine website."
            )
            advice = "That said, always be careful — only share personal details on websites you fully trust."
        else:
            reason = (
                f"The website <b>{domain}</b> looks mostly fine, though there are "
                f"a couple of small things worth knowing about this link."
            )
            advice = (
                "Our AI thinks it's probably safe, but since you're not 100% sure, "
                "double-check the website name carefully before entering anything personal."
            )
        message = (
            f"✅ <b>This link looks safe.</b><br><br>"
            f"{reason}<br><br>"
            f"{advice}"
        )
        box_style = (
            "background:rgba(5,46,22,0.92);border:2px solid #22c55e;border-radius:14px;"
            "padding:1.4rem 1.7rem;color:#86efac;margin-top:1.2rem;line-height:2;"
            "font-size:0.95rem;backdrop-filter:blur(10px);"
        )

    return f'<div style="{box_style}">{message}</div>'

# ══════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════
st.markdown("<div class='main-title'> Phishing URL Detector</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='main-subtitle'>Not sure if a link is safe? Paste it below and we'll check it for you.</div>",
    unsafe_allow_html=True
)

model, tokenizers, seg_max = load_artifacts()

if model is None:
    st.markdown("""
    <div style='background:rgba(69,10,10,0.9);border:2px solid #ef4444;border-radius:12px;
                padding:1.5rem;color:#fca5a5;margin:1rem 0;backdrop-filter:blur(10px);'>
    <b>⛔ Model files not found.</b><br><br>
    Please make sure these files are in a folder called <code>models/</code>
    next to this app:<br><br>
    <code>models/best_model.h5</code><br>
    <code>models/tokenizers.pkl</code><br>
    <code>models/seg_max.pkl</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── URL Input ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
url_in = st.text_input(
    "🔗 Paste the link you want to check",
    placeholder="e.g. https://paypal.secure-login.xyz/verify?id=123",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    go = st.button("🔍 Check this link", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Result ─────────────────────────────────────────────────────────────────
if go and url_in.strip():
    with st.spinner("🔍 Checking the link…"):
        try:
            result = predict(url_in.strip(), model, tokenizers, seg_max)
        except Exception as e:
            st.markdown(f"""
            <div style='background:rgba(69,10,10,0.9);border:2px solid #ef4444;
                        border-radius:12px;padding:1.2rem;color:#fca5a5;
                        backdrop-filter:blur(10px);'>
            ❌ <b>Something went wrong while checking:</b><br><code>{str(e)}</code>
            </div>""", unsafe_allow_html=True)
            st.stop()

    ph    = result["phishing_prob"]
    label = result["label"]
    conf  = result["confidence"]
    segs  = result["segments"]
    flags = risk_flags(url_in.strip(), segs)

    # ── 1. Verdict card ───────────────────────────────────────────────────
    box_class = "result-phishing" if label == "Phishing" else "result-legit"
    icon      = "🚨" if label == "Phishing" else "✅"
    verdict   = "Dangerous Link" if label == "Phishing" else "Looks Safe"
    st.markdown(f"""
    <div class="{box_class}">
        <div class="result-title">{icon} {verdict}</div>
        <div class="result-stats">
            We are <b>{conf*100:.0f}% confident</b> in this result &nbsp;|&nbsp;
            Link checked: <code style='font-size:0.8rem'>{url_in.strip()[:80]}{'…' if len(url_in)>80 else ''}</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. Safety meter ───────────────────────────────────────────────────
    bar_color = "#ef4444" if label == "Phishing" else "#22c55e"
    safety    = (1 - ph) * 100
    st.markdown(f"""
    <div style='margin:0.5rem 0 1.4rem 0;'>
        <div style='display:flex;justify-content:space-between;
                    color:#9ca3af;font-size:0.82rem;margin-bottom:6px;'>
            <span>🔴 Dangerous</span>
            <span style='color:#c4b5fd;font-weight:600;'>Safety Meter</span>
            <span>🟢 Safe</span>
        </div>
        <div style='background:rgba(30,27,75,0.8);border-radius:999px;height:14px;overflow:hidden;'>
            <div style='width:{safety:.1f}%;background:{bar_color};
                        height:100%;border-radius:999px;
                        box-shadow:0 0 12px {bar_color};'></div>
        </div>
        <div style='text-align:center;color:#9ca3af;font-size:0.78rem;margin-top:5px;'>
            Safety score: {safety:.0f} / 100
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 3. What we found ──────────────────────────────────────────────────
    col_seg, col_flags = st.columns(2)

    with col_seg:
        st.markdown(
            "<span style='color:#a78bfa;font-weight:700;'>🔎 What's inside this link</span>",
            unsafe_allow_html=True
        )
        seg_labels = {
            "subdomain" : "Prefix",
            "domain_tld": "Website name",
            "path"      : "Page",
            "query"     : "Extra info",
        }
        for seg, lbl in seg_labels.items():
            val = segs[seg] if segs[seg] != "none" else "<i style='color:#6b7280'>none</i>"
            st.markdown(
                f'<div class="seg-card"><span class="seg-label">{lbl}:</span> {val}</div>',
                unsafe_allow_html=True
            )

    with col_flags:
        st.markdown(
            "<span style='color:#a78bfa;font-weight:700;'>🚩 Things we noticed</span>",
            unsafe_allow_html=True
        )
        if flags:
            flags_html = "<br>".join(flags)
            st.markdown(f'<div class="flag-box">{flags_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="flag-safe">👍 Nothing suspicious found in the link structure.</div>',
                unsafe_allow_html=True
            )

    # ── 4. Plain-English explanation ──────────────────────────────────────
    st.markdown(build_explanation(label, ph, segs, flags), unsafe_allow_html=True)

elif go:
    st.markdown("""
    <div style='background:rgba(28,25,23,0.9);border:1px solid #78350f;border-radius:10px;
                padding:0.8rem 1rem;color:#fde68a;text-align:center;backdrop-filter:blur(8px);'>
    ⚠️ Please paste a link first before clicking Check.
    </div>""", unsafe_allow_html=True)

# ── Demo Examples ──────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<span style='color:#a78bfa;font-weight:700;font-size:0.95rem;'>⚡ Try an example</span>",
    unsafe_allow_html=True
)

examples = [
    ("✅ Real Google link",       "https://www.google.com/search?q=weather"),
    ("🚨 Fake PayPal link",       "http://paypal.secure-login.verify-account.xyz/update?user=123"),
    ("🚨 Number-only fake link",  "http://192.168.0.1/admin/signin.php?redirect=%2Fdashboard"),
    ("🚨 Fake Amazon link",       "http://amazon.account-suspended.com/verify/identity?id=99"),
]

cols = st.columns(len(examples))
for col, (ex_label, ex_url) in zip(cols, examples):
    with col:
        if st.button(ex_label, key=ex_url, use_container_width=True):
            with st.spinner("Checking…"):
                try:
                    r       = predict(ex_url, model, tokenizers, seg_max)
                    verdict = "🚨 Dangerous" if r["phishing_prob"] >= 0.5 else "✅ Safe"
                    color   = "rgba(127,29,29,0.9)" if "Dangerous" in verdict else "rgba(5,46,22,0.9)"
                    border  = "#ef4444" if "Dangerous" in verdict else "#22c55e"
                    st.markdown(f"""
                    <div style='background:{color};border:1px solid {border};
                                border-radius:10px;padding:0.7rem;
                                font-size:0.82rem;color:#f3f4f6;margin-top:0.4rem;
                                backdrop-filter:blur(8px);'>
                    <b>{verdict}</b><br>
                    We are {(1 - r['phishing_prob'] if r['phishing_prob'] < 0.5 else r['phishing_prob'])*100:.0f}%
                    confident<br>
                    <code style='font-size:0.72rem'>{ex_url[:45]}…</code>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(str(e))

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>Phishing URL Detector &nbsp;|&nbsp; Powered by AI &nbsp;|&nbsp; Built with Streamlit</div>",
    unsafe_allow_html=True
)