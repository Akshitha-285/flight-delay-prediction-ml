# src/predict.py

import joblib
import numpy as np

MODEL_PATH = "models/random_forest_model.pkl"

def predict_delay(input_features):
    model = joblib.load(MODEL_PATH)

    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        return "Flight will be delayed."
    else:
        return "Flight will arrive on time."


if __name__ == "__main__":
    sample_input = [15, 3, 19805, 10397, 13930, 1430, 1600, 0, 0, 800]
    result = predict_delay(sample_input)
    print(result)