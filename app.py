from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

# ========== CONFIG ========== #
IMAGE_SIZE = (224, 224)
class_names = ['Healthy', 'Tea leaf blight', 'Tea red leaf spot', 'Tea red scab']
model_path = "models/percobaan_baru_1.tflite"

# ========== INIT FASTAPI ========== #
app = FastAPI()

# Optional: Biar bisa diakses dari frontend mana saja (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ubah kalau mau lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== LOAD TFLITE MODEL ========== #
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ========== PREPROCESS FUNCTION ========== #
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img).astype(np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ========== PREDICT FUNCTION ========== #
def predict(image_bytes):
    input_data = preprocess_image(image_bytes)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction = output_data[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = round(float(np.max(prediction)) * 100, 2)

    return predicted_class, confidence

# ========== ROUTES ========== #
@app.get("/")
def home():
    return {"message": "Tea Leaf Disease Prediction API (TFLite)"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        predicted_class, confidence = predict(contents)
        return JSONResponse({
            "class": predicted_class,
            "confidence": f"{confidence}%",
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
