from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()

print("[INFO] Mulai load TFLite model...")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="percobaan_baru_1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("[INFO] Model TFLite berhasil dimuat.")
print("Input shape yang diharapkan:", input_details[0]['shape'])

class_names = ['Healthy', 'Tea leaf blight', 'Tea red leaf spot', 'Tea red scab']

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        print("[INFO] Menerima permintaan prediksi...")

        # Baca gambar dari request
        contents = await image.read()
        print(f"[INFO] Ukuran file yang diterima: {len(contents)} bytes")

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32)

        print("[INFO] Gambar berhasil dibaca dan diubah ke array. Shape:", img.shape)

        img = tf.keras.applications.efficientnet.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        print("[INFO] Melakukan inferensi...")

        # Prediksi
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_idx = np.argmax(output)
        pred_class = class_names[pred_idx]
        confidence = round(float(np.max(output)) * 100, 2)

        print("[INFO] Prediksi selesai. Kelas:", pred_class, "| Confidence:", confidence)

        return JSONResponse({
            "status": "success",
            "prediction": pred_class,
            "confidence": f"{confidence}%"
        })

    except Exception as e:
        print("[ERROR] Terjadi kesalahan:", str(e))
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
