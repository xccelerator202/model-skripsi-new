from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="percobaan_baru_1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['Healthy', 'Tea leaf blight', 'Tea red leaf spot', 'Tea red scab']

# Endpoint
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Baca gambar dari request
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        # Prediksi
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_idx = np.argmax(output)
        pred_class = class_names[pred_idx]
        confidence = round(float(np.max(output)) * 100, 2)

        return JSONResponse({
            "status": "success",
            "prediction": pred_class,
            "confidence": f"{confidence}%"
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
