# Defect Detection API  

This project is a FastAPI-based web service for detecting defects in images using a CNN model. The model is trained to classify images as **Defective** or **Non-Defective**.  

## Features  
- **FastAPI** for serving the model  
- **PyTorch** for loading and running the CNN model  
- **Image preprocessing** for model compatibility  
- **API endpoint** for image classification  

## Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/defect-detection-api.git
   cd defect-detection-api
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:  
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. Open the API docs in your browser:  
   ```
   http://127.0.0.1:8000/docs
   ```
