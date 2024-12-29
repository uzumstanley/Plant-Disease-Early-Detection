from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras import models, layers
from keras.models import Sequential
MODEL = tf.keras.models.load_model("../models/2.keras")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
app = FastAPI()

@app.get("/ping")
async def ping():
    return "hello world"
#data parameter is justbytes read from uploaded file(image)
def read_file_as_image(data) -> np.ndarray:
    return np.array(Image.open(BytesIO(data)))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    #store file as bytes
    image = read_file_as_image(await file.read())
    #converting image into a batch as our model only accepts batch
    image_batch = np.expand_dims(image, 0)
    #converting image_batch which is a numpy array into a tensor
   
    prediction = MODEL.predict(image_batch)
    #prediction is a batch and in batch there's only one image
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
  
    return {
        
        'class': predicted_class,
        'confidence': float(confidence)
    }
origins = [
    "http://localhost",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins= origins,
    allow_credentials =True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if __name__ == "__main__":
    uvicorn.run(app,host = 'localhost', port = 8000)