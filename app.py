import pickle
from flask import Flask, render_template, request
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

import torch
from torchvision import models

app = Flask(__name__)

# Load the trained model from the pickle file
model = models.alexnet(pretrained = True)  # Replace 'YourModel' with your actual model architecture
model.load_state_dict(torch.load("your_model.pkl"))
model.eval()

# Define transformations for incoming images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define a route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image from the form
        uploaded_image = request.files['image']
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            image = transform(image)
            image = image.unsqueeze(0)  # Add a batch dimension

            with torch.no_grad():
                outputs = model(image)
                _, prediction = torch.max(outputs, 1)
                class_index = prediction.item()
                classes = ['Pebbles', 'Shells']
                predicted_class = classes[class_index]
                
            return render_template('index.html', prediction=predicted_class)

    except Exception as e:
        print(f"Exception: {e}")
        return render_template('index.html', error_message=str(e))

if __name__ == '__main':
    app.run(debug=True, host='0.0.0.0', port=8080)
