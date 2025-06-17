import io
import pickle
import librosa
import numpy as np
from flask import Flask, request, jsonify

# Custom startup message
print("\n" + "="*40)
print("   Baby Cry Prediction Server")
#print("   By: RP")
print("="*40 + "\n")

# Initialize Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = "cry_xgboost_model.pkl"

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# Ensure the loaded model is correct
if isinstance(model, tuple):  
    model = model[0]  # Extract actual model from tuple

# Constants
SAMPLE_RATE = 16000

# Condition Mapping
condition_mapping = {
    0: "Asphyxia",
    1: "Deaf",
    2: "Hunger",
    3: "Normal",
    4: "Pain"
}

# Flask API Route
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    try:
        # Read audio file
        file = request.files["file"]
        audio_data = file.read()

        # Load audio and extract MFCC
        audio, _ = librosa.load(io.BytesIO(audio_data), sr=SAMPLE_RATE)
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)

        # Make prediction
        predicted_label = int(model.predict(mfcc_mean)[0])
        condition = condition_mapping.get(predicted_label, "Unknown")

        return jsonify({"condition": condition})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
