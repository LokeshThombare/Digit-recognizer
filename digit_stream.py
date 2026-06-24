import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Digit Recognizer")
st.write("Upload an image or draw a digit to get a prediction.")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mnist_model.keras')  # FIX: was tf.keras.load_model
    return model

model = load_model()  # FIX: function was defined but never called

# FIX: st.tabs() takes a list, not two separate string arguments
tab1, tab2 = st.tabs(["📁 Upload Image", "✏️ Draw a Digit"])

with tab1:
    st.header("Upload a File")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])  # FIX: "jepg" typo

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")  # FIX: was image.open (lowercase)
        image = image.resize((28, 28))                  # FIX: resize needs a tuple (28,28)

        st.image(uploaded_file, width=150, caption="Your uploaded image")

        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 784)

        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)          # FIX: consistent variable name
        confidence = prediction[0][predicted_digit] * 100  # FIX: was never defined

        st.success(f"Predicted Digit: **{predicted_digit}**")  # FIX: was st.sucess (typo)
        st.write(f"Confidence: {confidence:.2f}%")

with tab2:
    st.header("Draw a Digit")
    st.write("Draw any digit from 0 to 9 in the box below")

    canvas = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("Predict"):
        if canvas.image_data is not None:
            img = canvas.image_data[:, :, 0]
            img = Image.fromarray(img.astype('uint8')).resize((28, 28))  # FIX: tuple for resize

            img_array = np.array(img) / 255.0   # FIX: was np.array(image) — wrong variable
            img_array = img_array.reshape(1, 784)

            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
            confidence = prediction[0][predicted_digit] * 100

            st.success(f"Predicted Digit: **{predicted_digit}**")
            st.write(f"Confidence: {confidence:.2f}%")

            st.write("Confidence per digit:")
            for digit in range(10):
                bar = '█' * int(prediction[0][digit] * 50)  # FIX: was using wrong index variable
                st.write(f"{digit}: {prediction[0][digit]:.4f} {bar}")
