import os
from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models

app = Flask(__name__)

# Load your pre-trained model
model = models.alexnet(pretrained=True)

model.load_state_dict(torch.load("your_model.pkl"))
model.eval()

# Define a list of class labels
classes = ['Pebbles', 'Shells']  # Replace with your class labels

# Set the path to the uploads folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Define a route to render the HTML form

@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle form submission and image classification

@app.route('/predict', methods=['POST'])

def predict():
    try:
        # Get the uploaded image from the form
        uploaded_image = request.files['image']
        if uploaded_image:
            # Save the uploaded image to the uploads folder
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image.filename)
            uploaded_image.save(image_path)

            # Predict the uploaded image
            predicted_class = predict_image(image_path, model, transform, classes)

            return render_template('index.html', prediction=predicted_class, image_path=image_path)

    except Exception as e:
        print(f"Exception: {e}")
        return render_template('index.html', error_message=str(e))

def predict_image(image_path, model, transform, classes):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension

    # Make a prediction
    with torch.no_grad():
        outputs = model(image)
        _, prediction = torch.max(outputs, 1)
        predicted_class = classes[prediction.item()]

    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)