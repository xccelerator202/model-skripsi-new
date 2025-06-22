from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="percobaan_baru_1.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels
class_names = ['Healthy', 'Tea leaf blight', 'Tea red leaf spot', 'Tea red scab']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img, dtype=np.float32)

        # Gunakan preprocessing dari EfficientNet agar sama dengan training
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_idx = int(np.argmax(output_data))
        pred_class = class_names[pred_idx]
        confidence = round(float(np.max(output_data)) * 100, 2)

        return jsonify({
            'status': 'success',
            'prediction': pred_class,
            'confidence': f"{confidence}%"
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
