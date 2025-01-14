import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps     
import mlflow
import numpy as np
from mlflow import MlflowClient


trackinguri = "http://127.0.0.1:5000/"

mlflow.set_tracking_uri(trackinguri)
client = MlflowClient(tracking_uri=trackinguri)

# Load the model
model_name = "MNIST-Digit-Recognizer"
stage = "Production"

# Get the latest version of the model in the specified stage
latest_versions = client.get_latest_versions(name=model_name, stages=[stage])

# Print the model name and version
for version in latest_versions:
    print(f"\nModel Name: {version.name}, Version: {version.version}\n")
    
model_uri = f"models:/{model_name}/{stage}"

Dependencis = mlflow.pyfunc.get_model_dependencies(model_uri)

print("\nDigit Recognizer\n")

model = mlflow.pyfunc.load_model(model_uri)

print("\nDigit Recognizer\n")

st.title("Digit Recognizer")

# Create a canvas component
canvas_result = st_canvas(
    fill_color='#FFFFFF',
    stroke_width=20,
    stroke_color='#000000',
    background_color='#FFFFFF',
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button('Predict'):
    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        img = canvas_result.image_data
        
        input_image = Image.fromarray(img.astype('uint8'),'RGBA')
        input_image.save('img.png')
        img = Image.open("img.png")
        
        image = ImageOps.grayscale(img)
        image = ImageOps.invert(image)
        img = image.resize((28,28))
        img = np.array(img, dtype='float32')
        img = img/255
        img = img.reshape((1,1,28,28)) 
        
        pred = model.predict(image)
        # Get the max probability
        max_prob = np.max(pred)
        # Get the index of the max probability
        max_prob_index = np.argmax(pred)
        
        
        st.write(f"**Predicted Digit : {max_prob_index}**")
        st.write(f"**Confidence : {max_prob}**")
