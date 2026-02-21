import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    # Selecting important features as per project document
    selected_features = [
        'DAY_OF_MONTH',
        'DAY_OF_WEEK',
        'OP_CARRIER_AIRLINE_ID',
        'ORIGIN_AIRPORT_ID',
        'DEST_AIRPORT_ID',
        'DEP_TIME',
        'ARR_TIME',
        'DEP_DEL15',
        'DIVERTED',
        'DISTANCE',
        'ARR_DEL15'
    ]

    df = df[selected_features]

    # Remove missing values
    df = df.dropna()

    # Features and Label
    X = df.drop('ARR_DEL15', axis=1)
    y = df['ARR_DEL15']

    return X, y