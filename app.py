import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

MODEL_PATH = "results/car_detector.keras"

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH, compile=False)

model = load_my_model()

st.set_page_config(page_title="Car Detection AI", page_icon="🚗")
st.title("🚗 Car Object Detection AI")
st.write("Upload an image and the AI will draw a bounding box around detected cars.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image_rgb.shape

    # 2. Preprocess
    img_input = cv2.resize(image_rgb, (128, 128))
    img_input = img_to_array(img_input) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # confidence threshold (user-adjustable)
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)

    # 3. Predict
    pred_class, pred_box = model.predict(img_input)
    confidence = float(pred_class[0][0])
    ymin, xmin, ymax, xmax = pred_box[0]

    # Clamp box predictions to [0,1] to avoid weird values
    ymin = float(np.clip(ymin, 0.0, 1.0))
    xmin = float(np.clip(xmin, 0.0, 1.0))
    ymax = float(np.clip(ymax, 0.0, 1.0))
    xmax = float(np.clip(xmax, 0.0, 1.0))

    # 4. Denormalize
    x1, y1 = int(xmin * w), int(ymin * h)
    x2, y2 = int(xmax * w), int(ymax * h)
    
    # Ensure coordinates are valid and have sensible area
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    width = x2 - x1
    height = y2 - y1
    area_ratio = (width * height) / float(w * h) if width > 0 and height > 0 else 0.0

    # 5. Display
    if confidence > confidence_threshold and area_ratio > 0.001 and width > 5 and height > 5:
        display_img = image_rgb.copy()
        # Draw rectangle with adjusted coordinates
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
        
        st.image(display_img, caption="Processed Image", use_container_width=True)
        st.success(f"🚗 Car detected with {confidence*100:.2f}% confidence!")
    else:
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)
        st.warning(f"❌ No car detected (Confidence: {confidence*100:.2f}%)")

    with st.expander("See Raw Model Output"):
        st.write(f"Model Raw Output: {pred_box[0]}")
        st.write(f"Mapped Pixels: x1={x1}, y1={y1}, x2={x2}, y2={y2}")