from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

MODEL = tf.keras.models.load_model("models/potato_disease")  # path to the folder
CLASS_NAME = ['Early Blight', 'Late Blight', 'Healthy']

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend URL once deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_file_as_image(data) -> np.array:
    return np.array(Image.open(BytesIO(data)))

@app.get("/ping")
async def ping():
    return "Hello server is Live"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    prediction = MODEL.predict(img_batch)
    confidence = float(np.max(prediction[0]))
    return {"class": CLASS_NAME[np.argmax(prediction[0])], "confidence": confidence}