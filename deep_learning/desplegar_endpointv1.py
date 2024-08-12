from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# Load the model
model = load_model('modelophxv0.keras')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Verifica que 'val_dataset' y 'max_idx' existan en los datos recibidos
    if 'val_dataset' not in data or 'max_idx' not in data:
        return jsonify({"error": "Datos incompletos"}), 400

    try:
        # Convert the list back to a NumPy array
        val_dataset = np.array(data['val_dataset'])
        max_idx = data['max_idx']
        
        if max_idx >= val_dataset.shape[0]:
            return jsonify({"error": f"Index {max_idx} is out of bounds for axis 0 with size {val_dataset.shape[0]}"}), 400

        selected_sequence = val_dataset[max_idx]

        # Take the first 4 frames for the original sequence
        frames = selected_sequence[:4, ...]

        # Use the fifth frame from the original dataset
        fifth_frame = selected_sequence[4, ...]

        # Combine the first 4 frames with the fifth frame
        frames = np.concatenate((frames, np.expand_dims(fifth_frame, axis=0)), axis=0)

        # Predict the next frame
        new_prediction = model.predict(np.expand_dims(frames, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.squeeze(new_prediction[-1, ...], axis=-1)  # Remove the last dimension

        # Ensure dimensions match before concatenation
        print("frames.shape:", frames.shape)
        print("predicted_frame.shape:", predicted_frame.shape)
        
        if frames.shape[1:] != predicted_frame.shape:
            return jsonify({"error": f"Dimension mismatch: frames.shape = {frames.shape}, predicted_frame.shape = {predicted_frame.shape}"}), 500

        # Binarize the predicted frame to contain values of 0 or 1
        binary_predicted_frame = (predicted_frame > 0.5).astype(int)

        # Convert the binary predicted frame to a list
        binary_predicted_frame_list = binary_predicted_frame.tolist()

        return jsonify({"message": "Prediction complete.", "predicted_frame": binary_predicted_frame_list})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/files/<path:filename>')
def download_file(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True)
