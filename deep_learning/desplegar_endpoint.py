from flask import Flask, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('modelophxv0.keras')

# Define your custom colormap
colors = [(0.8, 0.8, 0.8), (1, 0.5, 0)]  # Light Gray to Orange
n_bins = 100  # Number of bins
cmap_name = 'custom1'
custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

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

        # Extend the set of prediction frames
        frames = np.concatenate((frames, np.expand_dims(predicted_frame, axis=0)), axis=0)

        # Create a new figure with custom grid layout
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 5)

        # First row for the original frames
        axes_original = [plt.subplot(gs[i]) for i in range(5)]
        for i, ax in enumerate(axes_original):
            ax.imshow(np.squeeze(frames[i]), cmap=custom_cmap, vmin=0, vmax=1)
            ax.set_title(f'Original Time {i+1}')

        # Second row for the "predicted" frames
        axes_predicted = [plt.subplot(gs[i+5]) for i in range(5)]
        for i, ax in enumerate(axes_predicted):
            cmap_choice = custom_cmap if i != 4 else 'viridis'
            frame_choice = np.squeeze(frames[i]) if i != 4 else np.squeeze(np.where(frames[-1] > frames[-1].max() * .5, 1, 0))
            title_choice = 'Original Time' if i != 4 else 'Actual Prediction'
            ax.imshow(frame_choice, cmap=cmap_choice, vmin=0, vmax=1)
            ax.set_title(f'{title_choice} {i+1}')

        # Save the plot to a file
        fig.savefig('prediction.png')
        plt.close(fig)

        return jsonify({"message": "Prediction complete. Check the saved image 'prediction.png'."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

