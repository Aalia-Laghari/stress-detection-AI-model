import streamlit as st
import numpy as np
import pickle
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

st.set_page_config(page_title="MindEase — Stress Companion", page_icon="🧘‍♀️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #05050f; }
.stApp { background: #05050f; min-height: 100vh; overflow-x: hidden; }

.bg-orbs { position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }
.orb { position: absolute; border-radius: 50%; filter: blur(90px); opacity: 0.18; animation: drift 18s ease-in-out infinite alternate; }
.orb-1 { width: 520px; height: 520px; background: radial-gradient(circle, #6366f1, #4f46e5); top: -160px; left: -100px; animation-delay: 0s; }
.orb-2 { width: 400px; height: 400px; background: radial-gradient(circle, #a78bfa, #7c3aed); top: 30%; right: -120px; animation-delay: -6s; }
.orb-3 { width: 340px; height: 340px; background: radial-gradient(circle, #60a5fa, #2563eb); bottom: 5%; left: 10%; animation-delay: -12s; }
.orb-4 { width: 260px; height: 260px; background: radial-gradient(circle, #f472b6, #db2777); top: 55%; left: 40%; animation-delay: -4s; opacity: 0.1; }
@keyframes drift {
    0%   { transform: translate(0, 0) scale(1); }
    33%  { transform: translate(40px, -30px) scale(1.05); }
    66%  { transform: translate(-20px, 50px) scale(0.97); }
    100% { transform: translate(30px, 20px) scale(1.03); }
}

.hero { text-align: center; padding: 4rem 1rem 2rem; position: relative; z-index: 2; }
.hero-pill { display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.25); color: #a5b4fc; font-size: 0.68rem; font-weight: 500; letter-spacing: 0.2em; text-transform: uppercase; padding: 0.4rem 1.1rem; border-radius: 100px; margin-bottom: 2rem; backdrop-filter: blur(8px); }
.hero-pill::before { content: ''; width: 6px; height: 6px; background: #818cf8; border-radius: 50%; box-shadow: 0 0 8px #818cf8; animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.7); } }
.hero-title { font-family: 'Playfair Display', serif; font-size: clamp(3rem, 8vw, 5rem); font-weight: 700; line-height: 1; letter-spacing: -0.03em; margin-bottom: 0.5rem; background: linear-gradient(135deg, #e0e7ff 0%, #a5b4fc 40%, #818cf8 70%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-sub { font-size: 1rem; color: rgba(200,210,255,0.4); font-weight: 300; letter-spacing: 0.04em; margin-bottom: 0.8rem; }
.hero-authors { font-size: 0.72rem; color: rgba(148,163,184,0.35); letter-spacing: 0.06em; margin-bottom: 2rem; }
.feature-row { display: flex; gap: 0.6rem; flex-wrap: wrap; justify-content: center; margin-bottom: 2rem; }
.feature-pill { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 100px; padding: 0.35rem 0.9rem; font-size: 0.7rem; color: rgba(148,163,184,0.45); letter-spacing: 0.05em; }

.glass-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 24px; padding: 2rem; margin-bottom: 1.2rem; backdrop-filter: blur(20px); box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06); position: relative; z-index: 2; }
.glass-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(129,140,248,0.4), transparent); border-radius: 24px 24px 0 0; }
.card-header { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1.5rem; }
.card-icon { width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; flex-shrink: 0; }
.icon-purple { background: rgba(129,140,248,0.12); border: 1px solid rgba(129,140,248,0.2); }
.icon-blue   { background: rgba(96,165,250,0.12);  border: 1px solid rgba(96,165,250,0.2); }
.card-title { font-size: 0.7rem; font-weight: 500; letter-spacing: 0.2em; text-transform: uppercase; color: rgba(165,180,252,0.6); }

.fancy-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), rgba(167,139,250,0.3), transparent); margin: 2rem 0; position: relative; z-index: 2; }

.result-card { border-radius: 24px; padding: 2.5rem 2rem; text-align: center; position: relative; overflow: hidden; margin-bottom: 1.2rem; z-index: 2; }
.result-card-low    { background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(5,5,15,0.95) 60%); border: 1px solid rgba(16,185,129,0.2); box-shadow: 0 0 60px rgba(16,185,129,0.08), inset 0 1px 0 rgba(16,185,129,0.1); }
.result-card-medium { background: linear-gradient(135deg, rgba(234,179,8,0.08) 0%, rgba(5,5,15,0.95) 60%); border: 1px solid rgba(234,179,8,0.2); box-shadow: 0 0 60px rgba(234,179,8,0.08), inset 0 1px 0 rgba(234,179,8,0.1); }
.result-card-high   { background: linear-gradient(135deg, rgba(239,68,68,0.08) 0%, rgba(5,5,15,0.95) 60%); border: 1px solid rgba(239,68,68,0.2); box-shadow: 0 0 60px rgba(239,68,68,0.08), inset 0 1px 0 rgba(239,68,68,0.1); }
.result-glow { position: absolute; width: 300px; height: 300px; border-radius: 50%; filter: blur(80px); opacity: 0.12; top: -100px; right: -80px; pointer-events: none; }
.glow-low    { background: #10b981; }
.glow-medium { background: #eab308; }
.glow-high   { background: #ef4444; }
.result-emoji { font-size: 4rem; line-height: 1; margin-bottom: 1rem; display: block; animation: float-emoji 3s ease-in-out infinite; }
@keyframes float-emoji { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
.result-label { font-family: 'Playfair Display', serif; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0.5rem; }
.label-low    { color: #34d399; }
.label-medium { color: #fbbf24; }
.label-high   { color: #f87171; }
.result-desc { font-size: 0.88rem; color: rgba(200,210,255,0.4); font-weight: 300; letter-spacing: 0.02em; }

.bar-wrap { margin: 2rem 0 0.5rem; }
.bar-track { height: 10px; border-radius: 100px; background: rgba(255,255,255,0.06); position: relative; overflow: visible; border: 1px solid rgba(255,255,255,0.05); }
.bar-fill { height: 100%; border-radius: 100px; position: relative; }
.fill-low    { background: linear-gradient(90deg, #059669, #34d399); box-shadow: 0 0 16px rgba(52,211,153,0.5); }
.fill-medium { background: linear-gradient(90deg, #d97706, #fbbf24); box-shadow: 0 0 16px rgba(251,191,36,0.5); }
.fill-high   { background: linear-gradient(90deg, #dc2626, #f87171); box-shadow: 0 0 16px rgba(248,113,113,0.5); }
.bar-thumb { width: 22px; height: 22px; border-radius: 50%; background: white; position: absolute; right: -11px; top: 50%; transform: translateY(-50%); box-shadow: 0 0 16px rgba(255,255,255,0.6), 0 0 0 4px rgba(255,255,255,0.15); }
.bar-labels { display: flex; justify-content: space-between; margin-top: 0.7rem; font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase; color: rgba(148,163,184,0.35); }

.quote-wrap { background: rgba(129,140,248,0.04); border: 1px solid rgba(129,140,248,0.1); border-left: 3px solid rgba(129,140,248,0.5); border-radius: 0 12px 12px 0; padding: 1rem 1.4rem; margin: 1.2rem 0; font-style: italic; font-size: 0.9rem; line-height: 1.7; color: rgba(200,210,255,0.55); position: relative; z-index: 2; }
.music-card { background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.15); border-radius: 14px; padding: 1rem 1.4rem; display: flex; align-items: center; gap: 0.8rem; margin: 0.8rem 0; position: relative; z-index: 2; }
.music-icon { width: 40px; height: 40px; background: linear-gradient(135deg, #4f46e5, #7c3aed); border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; flex-shrink: 0; box-shadow: 0 4px 16px rgba(79,70,229,0.35); }
.music-text { font-size: 0.82rem; color: rgba(165,180,252,0.7); }
.music-text a { color: #a5b4fc; text-decoration: none; font-weight: 500; }
.tip-box { background: linear-gradient(135deg, rgba(99,102,241,0.06), rgba(167,139,250,0.06)); border: 1px solid rgba(129,140,248,0.12); border-radius: 14px; padding: 1rem 1.4rem; color: rgba(165,180,252,0.6); font-size: 0.85rem; line-height: 1.6; text-align: center; margin-top: 1rem; position: relative; z-index: 2; }
.footer { text-align: center; padding: 2.5rem 0 1.5rem; color: rgba(148,163,184,0.2); font-size: 0.7rem; letter-spacing: 0.06em; position: relative; z-index: 2; }

.stTextInput > div > div > input, .stNumberInput > div > div > input { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 12px !important; color: rgba(220,230,255,0.85) !important; font-family: 'Inter', sans-serif !important; font-size: 0.88rem !important; }
.stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus { border-color: rgba(129,140,248,0.45) !important; box-shadow: 0 0 0 3px rgba(129,140,248,0.07) !important; background: rgba(99,102,241,0.04) !important; }
.stTextInput label, .stNumberInput label { color: rgba(165,180,252,0.55) !important; font-size: 0.82rem !important; font-weight: 400 !important; }
.stButton > button { background: linear-gradient(135deg, #4338ca, #6d28d9, #7c3aed) !important; color: white !important; border: none !important; border-radius: 14px !important; padding: 0.85rem 2rem !important; font-family: 'Inter', sans-serif !important; font-size: 0.88rem !important; font-weight: 500 !important; letter-spacing: 0.06em !important; width: 100% !important; box-shadow: 0 4px 32px rgba(109,40,217,0.4), inset 0 1px 0 rgba(255,255,255,0.1) !important; }
</style>

<div class="bg-orbs">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
    <div class="orb orb-4"></div>
</div>
""", unsafe_allow_html=True)


HF_MODEL = "Aalia-Laghari/mindease-stress-model"

@st.cache_resource
def load_models():
    sensor_model = pickle.load(open("stress_trained.sav", "rb"))
    scaler = pickle.load(open("scaler.sav", "rb"))
    text_tokenizer = DistilBertTokenizerFast.from_pretrained(HF_MODEL)
    text_model = DistilBertForSequenceClassification.from_pretrained(
        HF_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    text_model.eval()
    return sensor_model, scaler, text_model, text_tokenizer

sensor_model, scaler, text_model, text_tokenizer = load_models()


def predict_stress_from_sensor(sensor_data):
    sensor_data = np.asarray(sensor_data).reshape(1, -1)
    sensor_data = scaler.transform(sensor_data)
    pred_class = sensor_model.predict(sensor_data)[0]
    return pred_class / 2.0


def predict_stress_from_text(text):
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = text_model(**inputs)
        pred_class = torch.argmax(outputs.logits, dim=1).item()
    return pred_class / 2.0


quotes = {
    "high": [
        "Even the darkest night will end and the sun will rise. — Victor Hugo",
        "Stress is not what happens to us. It's our response to what happens. — Hans Selye",
    ],
    "medium": [
        "Don't be pushed by your problems, be led by your dreams. — Ralph Waldo Emerson",
        "Take rest; a field that has rested gives a bountiful crop. — Ovid",
    ],
    "low": [
        "Peace comes from within. Do not seek it without. — Buddha",
        "Happiness is not something ready-made. It comes from your own actions. — Dalai Lama",
    ],
}

music_links = {
    "high": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3",
    ],
    "medium": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3",
    ],
    "low": [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3",
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3",
    ],
}

st.markdown("""
<div class="hero">
    <div class="hero-pill">✦  AI Stress Detection System</div>
    <div class="hero-title">MindEase</div>
    <div class="hero-sub">Your personal stress companion — powered by AI</div>
    <div class="hero-authors">Aalia Laghari · 22CS014 &nbsp;·&nbsp; Lareb · 22CS066 &nbsp;·&nbsp; Syeda Reeba Zaidi · 22-21CS120</div>
    <div class="feature-row">
        <span class="feature-pill">🧠 NLP Analysis</span>
        <span class="feature-pill">📡 Sensor Fusion</span>
        <span class="feature-pill">🎵 Music Therapy</span>
        <span class="feature-pill">🌿 Mindfulness Tips</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <div class="card-header">
        <div class="card-icon icon-purple">💬</div>
        <div class="card-title">Reflective Questions</div>
    </div>
""", unsafe_allow_html=True)

q1 = st.text_input("How has your day been so far?", placeholder="Tell me about your day...")
q2 = st.text_input("Is there anything that's been on your mind lately?", placeholder="Any thoughts weighing on you?")
q3 = st.text_input("How's your energy right now — calm, restless, or tired?", placeholder="Describe your energy level...")
q4 = st.text_input("Have you felt more at ease or under pressure today?", placeholder="At ease, or under pressure?")
q5 = st.text_input("What's one thing that might make your day feel lighter?", placeholder="Something that brings you joy...")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
    <div class="card-header">
        <div class="card-icon icon-blue">📡</div>
        <div class="card-title">Body Sensor Data</div>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
with col2:
    temperature = st.number_input("🌡️ Body Temp (°C)", min_value=0.0, max_value=100.0, step=0.1)
with col3:
    steps = st.number_input("👣 Step Count", min_value=0, max_value=50000, step=1)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if st.button("✨  Analyze My Stress Level"):
    user_text = " ".join([q1, q2, q3, q4, q5])

    with st.spinner("Analyzing your responses..."):
        text_stress = predict_stress_from_text(user_text)
        sensor_stress = predict_stress_from_sensor([humidity, temperature, steps])
        final_score = (text_stress + sensor_stress) / 2

    if final_score > 0.6:
        level_key, level_text, level_emoji = "high", "High Stress", "😟"
        bar_fill_class, card_class, glow_class, label_class = "fill-high", "result-card-high", "glow-high", "label-high"
        bar_pct = 88
        desc = "Your stress levels are elevated. It's okay — let's take it one breath at a time."
    elif final_score > 0.3:
        level_key, level_text, level_emoji = "medium", "Medium Stress", "🙂"
        bar_fill_class, card_class, glow_class, label_class = "fill-medium", "result-card-medium", "glow-medium", "label-medium"
        bar_pct = 52
        desc = "You're managing, but there's some tension. A little rest will go a long way."
    else:
        level_key, level_text, level_emoji = "low", "Low Stress", "😊"
        bar_fill_class, card_class, glow_class, label_class = "fill-low", "result-card-low", "glow-low", "label-low"
        bar_pct = 14
        desc = "You're calm and balanced. Keep nurturing that inner peace."

    st.markdown(f"""
    <div class="fancy-divider"></div>
    <div class="result-card {card_class}">
        <div class="result-glow {glow_class}"></div>
        <span class="result-emoji">{level_emoji}</span>
        <div class="result-label {label_class}">{level_text}</div>
        <div class="result-desc">{desc}</div>
        <div class="bar-wrap">
            <div class="bar-track">
                <div class="bar-fill {bar_fill_class}" style="width:{bar_pct}%;">
                    <div class="bar-thumb"></div>
                </div>
            </div>
            <div class="bar-labels">
                <span>🟢 Low</span><span>🟡 Medium</span><span>🔴 High</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    quote = np.random.choice(quotes[level_key])
    music = np.random.choice(music_links[level_key])

    st.markdown(f'<div class="quote-wrap">💬 &nbsp;{quote}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="music-card">
        <div class="music-icon">🎵</div>
        <div class="music-text">Recommended music for your mood<br>
            <a href="{music}" target="_blank">▶ &nbsp;Click to listen</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.audio(music, format="audio/mp3")
    st.markdown('<div class="tip-box">🌿 &nbsp;Take a moment for yourself — stretch, breathe deeply, or step outside for a brief walk.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="fancy-divider" style="margin-top:3rem;"></div>
<div class="footer">© 2025 &nbsp;·&nbsp; AI Semester Project &nbsp;·&nbsp; Mehran University of Engineering & Technology</div>
""", unsafe_allow_html=True)
