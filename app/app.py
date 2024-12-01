import streamlit as st
from torch import device, cuda
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.util import get_class_of_sample


st.title("Simpsons Classification")

# Device configuration (CPU or GPU)
DEVICE = device("cuda" if cuda.is_available() else "cpu")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    pred = get_class_of_sample(image, DEVICE)[0]
    st.write(f"Predicted class: {pred}")
