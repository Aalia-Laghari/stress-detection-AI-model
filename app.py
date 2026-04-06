import streamlit as st
import numpy as np
import pickle
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

st.set_page_config(page_title="MindEase — Stress Companion", page_icon="🧘‍♀️", layout="centered")

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


def display_gradient_bar(score_10):
    percent = score_10 * 10
    st.markdown(
        f"""
        <div style="background: linear-gradient(to right, #4CAF50, #FFEB3B, #F44336);
                    border-radius: 10px; width: 100%; height: 25px; position: relative;">
            <div style="width:{percent}%; background-color:rgba(0,0,0,0.3); height: 100%;
                        border-radius: 10px; text-align:center;">
                <span style="position:absolute; left:50%; transform:translateX(-50%);
                             color:black; font-weight:bold;">
                    {score_10} / 10
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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

st.title("🧘‍♀️ MindEase — Stress Companion")
st.markdown("Take a moment 🌿 and answer these short questions — no rush.")

q1 = st.text_input("💬 How has your day been so far?")
q2 = st.text_input("💭 Is there anything that's been on your mind lately?")
q3 = st.text_input("⚡ How's your energy right now — calm, restless, or tired?")
q4 = st.text_input("😌 Have you felt more at ease or under pressure today?")
q5 = st.text_input("🌈 What's one thing that might make your day feel lighter?")

st.markdown("---")
st.subheader("Now let's add your body data")

humidity = st.number_input("💧 Humidity Value", min_value=0.0)
temperature = st.number_input("🌡️ Body Temperature (°C)", min_value=0.0)
steps = st.number_input("👣 Step Count", min_value=0)

if st.button("✨ Analyze My Stress Level"):
    user_text = " ".join([q1, q2, q3, q4, q5])

    text_stress = predict_stress_from_text(user_text)
    sensor_stress = predict_stress_from_sensor([humidity, temperature, steps])
    final_score = (text_stress + sensor_stress) / 2
    final_score_10 = round(final_score * 10, 1)

    st.markdown("---")
    st.subheader("🌤 Your Mood Check Summary")
    display_gradient_bar(final_score_10)

    if final_score > 0.6:
        level_key = "high"
        st.error(f"You seem stressed today. ({final_score_10}/10)")
    elif final_score > 0.3:
        level_key = "medium"
        st.warning(f"You're doing okay. ({final_score_10}/10)")
    else:
        level_key = "low"
        st.success(f"You're calm and balanced. ({final_score_10}/10)")

    quote = np.random.choice(quotes[level_key])
    music = np.random.choice(music_links[level_key])

    st.markdown(f"> 💬 *{quote}*")
    st.markdown(f"🎵 **Music Recommendation:** [Click to Listen]({music})")
    st.audio(music, format="audio/mp3")
    st.markdown("🌿 Take a moment for yourself — stretch, breathe, or enjoy a brief walk!")

st.markdown("---")
st.write("© 2025 | AI Semester Project — Mehran University of Engineering & Technology")
