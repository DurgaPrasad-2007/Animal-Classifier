from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import pickle
from datetime import datetime
import os

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'animal_classifier.h5')
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, 'class_indices.pkl')

# Load model and class indices
print("Loading model...")
try:
    print(f"Looking for model at: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

def preprocess_image(image):
    # Resize image
    image = image.resize((224, 224))
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
async def root():
    return {"message": "Animal Classification API is running"}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Get prediction
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get class name
        class_name = list(class_indices.keys())[predicted_class]
        
        # Create response
        response = {
            "label": class_name,
            "confidence": confidence,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "image_size": image.size,
                "model_version": "simple_cnn_v1"
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 