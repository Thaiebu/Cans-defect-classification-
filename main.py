from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import uvicorn
import os
from typing import List

# Define the model architecture
class DefectClassifier(nn.Module):
    def __init__(self):
        super(DefectClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32768, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize FastAPI app
app = FastAPI(title="Defect Classification API", 
              description="API for classifying images as defective or non-defective")

# Define transforms for inference (without augmentations)
inference_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Class names mapping
class_names = ['defect','normal']

# Global variables for model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

@app.on_event("startup")
async def load_model():
    global model
    # Initialize model
    model = DefectClassifier()
    
    # Check if model file exists
    model_path = "defect_detection_model_working.pth"  # Update with your model path
    if not os.path.exists(model_path):
        # For demo, notify that model file is missing
        print(f"Warning: Model file {model_path} not found. API will return dummy predictions.")
        return
    
    # Load model weights if file exists
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

def predict_single_image(image, model, transform):
    """Process a single image and return prediction"""
    # Ensure model is in eval mode
    model.eval()
    
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_name = class_names[predicted.item()]
    print(predicted)
    confidence_value = confidence.item()
    
    return {
        "prediction": class_name,
        "confidence": float(confidence_value)
    }

@app.post("/predict/", response_model=dict)
async def predict_image(file: UploadFile = File(...)):
    """
    Classify an image as defective or normal
    
    - **file**: Upload an image file
    
    Returns prediction and confidence score
    """
    # Check if model is loaded
    if model is None:
        return {"prediction": "dummy", "confidence": 0.5, "warning": "Model not loaded"}
    
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Make prediction
        result = predict_single_image(image, model, inference_transform)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Defect Classification API",
        "endpoints": {
            "/predict/": "Classify a single image",
            "/predict-batch/": "Classify multiple images"
        }
    }

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

# uvicorn main:app --reload