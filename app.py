import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

from utils.processing import prepare_image  # Make sure this import works correctly

st.title("Indonesian Food Classification")

st.write("""
This app uses a tensorflow machine learning model to classify images of Indonesian food.
The model has been trained on a dataset of Bakso, Rendang, Gado-gado, Sate, and Gudeg.
The model's accuracy is 97%.
""")

st.write("Upload an image of Indonesian food to classify it as Bakso, Rendang, Gado-gado, Sate, or Gudeg.")

# Load the model
@st.cache_resource
def load_classification_model():
    return load_model("model_acc_97.h5")

model = load_classification_model()

# Define class names
class_names = ["bakso", "gado", "gudeg", "rendang", "sate"]

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Prepare the image
    img_array = prepare_image(image)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    
    st.write(f"Prediction: {predicted_class}")
    
    # Display confidence scores
    st.write("Confidence Scores:")
    for i, score in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {score:.2f}")

if __name__ == '__main__':
    load_classification_model()