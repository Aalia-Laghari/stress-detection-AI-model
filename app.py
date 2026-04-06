import streamlit as st
import numpy as np
import pickle
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

st.set_page_config(page_title="MindEase — Stress Companion", page_icon="🧘‍♀️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 40%, #0a1020 100%); min-height: 100vh; }
.hero-container { text-align: center; padding: 3rem 1rem 1.5rem; }
.hero-badge { display: inline-block; background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3); color: #a5b4fc; font-size: 0.7rem; font-weight: 500; letter-spacing: 0.2em; text-transform: uppercase; padding: 0.35rem 1rem; border-radius: 100px; margin-bottom: 1.5rem; }
.hero-title { font-family: 'Cormorant Garamond', serif; font-size: 3.8rem; font-weight: 300; color: #f0f0ff; line-height: 1.1; margin: 0 0 0.5rem; letter-spacing: -0.02em; }
.hero-title span { background: linear-gradient(135deg, #818cf8, #a78bfa, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-authors { font-size: 0.78rem; font-weight: 400; color: rgba(165,180,252,0.7); letter-spacing: 0.08em; margin-bottom: 0.6rem; }
.hero-subtitle { font-size: 0.95rem; color: rgba(200,210,255,0.45); font-weight: 300; letter-spacing: 0.02em; margin-bottom: 2.5rem; }
.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.3), transparent); margin: 1.5rem 0; }
.section-label { font-size: 0.65rem; font-weight: 500; letter-spacing: 0.25em; text-transform: uppercase; color: rgba(165,180,252,0.5); margin-bottom: 1.2rem; margin-top: 2rem; }
.card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 1.8rem; margin-bottom: 1rem; }
.score-display { text-align: center; padding: 2rem 0 1rem; }
.score-number { font-family: 'Cormorant Garamond', serif; font-size: 5rem; font-weight: 300; line-height: 1; background: linear-gradient(135deg, #818cf8, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.score-label { font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase; color: rgba(165,180,252,0.5); margin-top: 0.3rem; }
.gradient-track { height: 8px; border-radius: 100px; background: linear-gradient(90deg, #22c55e, #eab308, #ef4444); margin: 1.2rem 0; position: relative; }
.gradient-thumb { width: 18px; height: 18px; border-radius: 50%; background: white; box-shadow: 0 0 12px rgba(165,180,252,0.8), 0 0 0 3px rgba(129,140,248,0.4); position: absolute; top: 50%; transform: translateY(-50%); }
.result-badge { display: inline-block; padding: 0.5rem 1.4rem; border-radius: 100px; font-size: 0.8rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.8rem; }
.badge-high { background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.3); color: #fca5a5; }
.badge-medium { background: rgba(234,179,8,0.12); border: 1px solid rgba(234,179,8,0.3); color: #fde047; }
.badge-low { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3); color: #86efac; }
.quote-block { border-left: 2px solid rgba(129,140,248,0.4); padding: 0.8rem 1.2rem; margin: 1.2rem 0; color: rgba(200,210,255,0.65); font-style: italic; font-size: 0.9rem; line-height: 1.6; }
.music-link { display: inline-flex; align-items: center; gap: 0.5rem; background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.25); color: #a5b4fc; padding: 0.5rem 1.2rem; border-radius: 100px; font-size: 0.8rem; text-decoration: none; margin-top: 0.5rem; }
.tip-box { background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.15); border-radius: 12px; padding: 0.9rem 1.2rem; color: rgba(165,180,252,0.7); font-size: 0.85rem; margin-top: 1rem; text-align: center; }
.footer { text-align: center; padding: 2rem 0 1rem; color: rgba(148,163,184,0.3); font-size: 0.72rem; letter-spacing: 0.05em; }
.stTextInput > div > div > input, .stNumberInput > div > div > input { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important; color: rgba(220,230,255,0.9) !important; font-family: 'DM Sans', sans-serif !important; }
.stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus { border-color: rgba(129,140,248,0.5) !important; box-shadow: 0 0 0 3px rgba(129,140,248,0.08) !important; }
.stTextInput label, .stNumberInput label { color: rgba(165,180,252,0.7) !important; font-size: 0.85rem !important; font-weight: 400 !important; }
.stButton > button { background: linear-gradient(135deg, #4f46e5, #7c3aed) !important; color: white !important; border: none !important; border-radius: 100px !important; padding: 0.75rem 2.5rem !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.85rem !important; font-weight: 500 !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; width: 100% !important; box-shadow: 0 4px 24px rgba(79,70,229,0.3) !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    sensor_model = pickle.load(open("stress_trained.sav", "rb"))
    text_model = DistilBertForSequenceClassification.from_pretrained("./final_stress_model")
    text_tokenizer = DistilBertTokenizerFast.from_pretrained("./final_stress_tokenizer")
    text_model.eval()
    return sensor_model, text_model, text_tokenizer

sensor_model, text_model, text_tokenizer = load_models()


def predict_stress_from_sensor(sensor_data):
    sensor_data = np.asarray(sensor_data).reshape(1, -1)
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
<div class="hero-container">
    <div class="hero-badge">✦ AI Stress Detection System</div>
    <div class="hero-title">🧘‍♀️ Mind<span>Ease</span></div>
    <div class="hero-title" style="font-size:1.6rem; margin-bottom:0.8rem;">— Stress Companion</div>
    <div class="hero-authors">Made By &nbsp;Aalia Laghari : 22CS014 &nbsp;·&nbsp; Lareb : 22CS066 &nbsp;·&nbsp; Syeda Reeba Zaidi : 22-21CS120</div>
    <div class="hero-subtitle">Take a moment 🌿 and answer these short questions — no rush.</div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">✦ Reflective Questions</div>', unsafe_allow_html=True)

q1 = st.text_input("💬 How has your day been so far?", placeholder="Tell me about your day...")
q2 = st.text_input("💭 Is there anything that's been on your mind lately?", placeholder="Any thoughts weighing on you?")
q3 = st.text_input("⚡ How's your energy right now — calm, restless, or tired?", placeholder="Describe your energy level...")
q4 = st.text_input("😌 Have you felt more at ease or under pressure today?", placeholder="At ease, or under pressure?")
q5 = st.text_input("🌈 What's one thing that might make your day feel lighter?", placeholder="Something that brings you joy...")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">✦ Body Data</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
with col2:
    temperature = st.number_input("🌡️ Body Temp (°C)", min_value=0.0, max_value=50.0, step=0.1)
with col3:
    steps = st.number_input("👣 Step Count", min_value=0, max_value=50000, step=1)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("✨ Analyze My Stress Level"):
    user_text = " ".join([q1, q2, q3, q4, q5])

    with st.spinner("Analyzing your responses..."):
        text_stress = predict_stress_from_text(user_text)
        sensor_stress = predict_stress_from_sensor([humidity, temperature, steps])
        final_score = (text_stress + sensor_stress) / 2
        final_score_10 = round(final_score * 10, 1)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">✦ Your Mood Check Summary</div>', unsafe_allow_html=True)

    if final_score > 0.6:
        level_key = "high"
        badge_class = "badge-high"
        level_text = "High Stress"
        level_emoji = "😟"
    elif final_score > 0.3:
        level_key = "medium"
        badge_class = "badge-medium"
        level_text = "Medium Stress"
        level_emoji = "🙂"
    else:
        level_key = "low"
        badge_class = "badge-low"
        level_text = "Low Stress"
        level_emoji = "😊"

    thumb_pos = min(max(final_score * 100, 2), 96)

    st.markdown(f"""
    <div class="card">
        <div class="score-display">
            <div class="score-number">{final_score_10}</div>
            <div class="score-label">Stress Score / 10</div>
            <div class="gradient-track">
                <div class="gradient-thumb" style="left: {thumb_pos}%;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; font-size:0.65rem;
                        color:rgba(148,163,184,0.4); letter-spacing:0.1em; text-transform:uppercase;">
                <span>Calm</span><span>Moderate</span><span>Stressed</span>
            </div>
            <br>
            <span class="result-badge {badge_class}">{level_emoji} {level_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    quote = np.random.choice(quotes[level_key])
    music = np.random.choice(music_links[level_key])

    st.markdown(f'<div class="quote-block">{quote}</div>', unsafe_allow_html=True)
    st.markdown(f'<a class="music-link" href="{music}" target="_blank">🎵 Listen to Recommended Music</a>', unsafe_allow_html=True)
    st.audio(music, format="audio/mp3")
    st.markdown('<div class="tip-box">🌿 Take a moment for yourself — stretch, breathe, or enjoy a brief walk.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="divider" style="margin-top: 3rem;"></div>
<div class="footer">© 2025 · AI Semester Project · Mehran University of Engineering & Technology</div>
""", unsafe_allow_html=True)
