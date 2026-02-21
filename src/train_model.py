# src/train_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def train_random_forest(X, y):

    # 80:20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Initialize Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

    # Classification Report
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # ----------------------------
    # 📊 Feature Importance
    # ----------------------------
    feature_importance = model.feature_importances_
    features = X.columns

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importance:\n")
    print(importance_df)

    plt.figure()
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance - Random Forest")
    plt.gca().invert_yaxis()
    plt.show()

    # ----------------------------
    # 📊 Confusion Matrix
    # ----------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return model


def save_model(model, path):
    joblib.dump(model, path)
    print(f"\nModel saved successfully at {path}")