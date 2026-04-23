import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array

MODEL_PATH = "results/car_detector.h5"

@st.cache_resource
def load_my_model():
    # Adding compile=False fixes the deserialization error for inference
    return load_model(MODEL_PATH, compile=False)

model = load_my_model()

st.title("🚗 Car Object Detection AI")
st.write("Upload an image and detect car with bounding box")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    orig = image.copy()

    # Preprocess image
    img = cv2.resize(image, (128, 128))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # -------- PREDICT --------
    # If your model has two outputs (classification and bounding box)
    pred_class, pred_box = model.predict(img)

    confidence = float(pred_class[0][0])
    box = pred_box[0]

    h, w, _ = orig.shape

    # Bounding box coordinates
    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)

    # -------- DRAW BOX --------
    if confidence > 0.5:
        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # -------- SHOW RESULTS --------
    st.image(orig, channels="BGR", caption="Detection Result")

    if confidence > 0.5:
        st.success(f"🚗 Car detected ({confidence*100:.2f}%)")
        st.info("The model detected vehicle-like features such as shape and structure.")
    else:
        st.error(f"❌ No car detected ({confidence*100:.2f}%)")
        st.warning("No strong car features found in this image.")