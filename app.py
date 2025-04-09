import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import folium
from streamlit_folium import st_folium
import gdown
import os

# التأكد من تحميل النموذج من Google Drive
model_path = "cnn_congestion_model.h5"
if not os.path.exists(model_path):
    with st.spinner("📥 جاري تحميل النموذج من Google Drive..."):
        url = "https://drive.google.com/file/d/1rczlSO7402EvyQnD_O-lNe6nQlE_YCA_"
        gdown.download(url, model_path, quiet=False)
        st.success("✅ تم تحميل النموذج!")

# تحميل النموذج
model = tf.keras.models.load_model(model_path)
class_names = ['خفيف', 'متوسط', 'عالي']

st.title("📸 تحليل مستوى الازدحام في الصورة")

uploaded_file = st.file_uploader("اختر صورة من التجمع", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((128, 128))
    st.image(image, caption="📷 الصورة المدخلة", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 3)

    prediction = model.predict(img_array)
    congestion = class_names[np.argmax(prediction)]

    st.subheader(f"🚦 مستوى الازدحام: **{congestion}**")

    st.subheader("📍 البوابات المتاحة:")

    gates = {
        "البوابة A": {"lat": 24.7840, "lon": 46.7265, "level": "خفيف"},
        "البوابة B": {"lat": 24.7832, "lon": 46.7282, "level": "عالي"},
        "البوابة C": {"lat": 24.7825, "lon": 46.7270, "level": "متوسط"},
    }

    m = folium.Map(location=[24.7838, 46.7270], zoom_start=17)
    color_map = {"خفيف": "green", "متوسط": "orange", "عالي": "red"}

    for name, info in gates.items():
        folium.Marker(
            location=[info["lat"], info["lon"]],
            popup=f"{name} - {info['level']}",
            icon=folium.Icon(color=color_map[info["level"]])
        ).add_to(m)

    st_folium(m, width=700, height=450)

    recommended_gate = next((g for g, v in gates.items() if v['level'] == 'خفيف'), None)
    if recommended_gate:
        st.success(f"✅ نوصي بالتوجه إلى: {recommended_gate}")
    else:
        st.warning("⚠️ جميع البوابات مزدحمة حالياً.")
