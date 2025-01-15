import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps     
import numpy as np
import torch
from torchvision import transforms
from loadmodel import getModel

import streamlit as st


@st.cache_resource
def load_model():
    if 'models' not in st.session_state:
        st.session_state.models = getModel()
        return st.session_state.models
    
    
if st.button("Refresh Models"):
    st.cache_resource.clear()

model = load_model()



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
        
        input_image = Image.fromarray(img.astype('uint8'), 'RGBA')
        input_image.save('img.png')
        img = Image.open("img.png")

        # Process the image
        image = ImageOps.grayscale(img)  # Convert to grayscale
        image = ImageOps.invert(image)   # Invert colors
        img = image.resize((28, 28))  # Resize to 28x28
        img = np.array(img, dtype='float32')
        img = img/255

        # Convert the image to a PyTorch tensor
        transform = transforms.ToTensor()  # Convert to tensor and scale pixel values to [0,1]
        image_tensor = transform(img)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)  # Shape: [1, 1, 28, 28]

        # Perform inference
        
        if model:
            with torch.no_grad():
                output = model(image_tensor)  # Forward pass
                predicted_label = torch.argmax(output, dim=1).item()  # Get predicted class
                confidence = output[0][predicted_label].item() * 100  # Confidence in percentage

                    
            
            st.write(f"**Predicted Digit : {predicted_label}**")
            st.write(f"**Confidence : {confidence}**")
        
        else :
            
            st.write(f"**No Model in Production..!!**")
            
            
