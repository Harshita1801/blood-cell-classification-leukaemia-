import streamlit as st
from utils.predict import predict_and_explain
from PIL import Image

# --- Background image CSS ---
import base64

def set_background_local(image_path):
    with open(image_path, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .block-container {{
            background-color: rgba(0, 0, 0, 0.4);
            padding: 2rem;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }}

        .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp div {{
            color: white;
        }}

        .stButton>button {{
            background-color: #ff4b4b;
            color: white;
            border-radius: 10px;
            font-weight: bold;
        }}

        .stFileUploader {{
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background_local(r"C:\Users\harsh\Downloads\Smoke Meets Blood Wallpaper By Helbazero.jpg")



st.title("🩸 Blood Pixels: Unraveling Leukemia with Deep Learning")

uploaded_file = st.file_uploader("Upload a blood cell image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("🔬 Diagnose"):
        pred_class, probability, lime_img = predict_and_explain(uploaded_file)

        st.markdown(f"### 🧪 Prediction: `{pred_class}`")
        st.markdown(f"**🧠 Confidence:** `{probability:.2f}%`")
        
        st.image(lime_img, caption="LIME Explanation", use_column_width=True)
