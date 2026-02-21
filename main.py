# main.py

from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_random_forest, save_model

DATA_PATH = "data/flight_data.csv"
MODEL_PATH = "models/random_forest_model.pkl"


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)

    print("Preprocessing data...")
    X, y = preprocess_data(df)

    print("Training Random Forest model...")
    model = train_random_forest(X, y)

    print("Saving trained model...")
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()