# ✈️ Flight Delay Prediction using Machine Learning

## 📌 Project Overview

This project predicts whether a flight will arrive late (arrival delay of 15 minutes or more) using Machine Learning techniques.

The model is trained on 583,000+ real-world flight records and analyzes important aviation-related features such as departure time, airport IDs, airline carrier, route distance, and more.


## 🎯 Problem Statement

Flight delays impact passengers, airlines, and airport operations.
The objective of this project is to build a classification model that predicts:

* **0 → On-Time Arrival**
* **1 → Delayed Arrival (15+ minutes)**


## 📊 Dataset Information

* Source: Kaggle – Flight Delay Prediction Dataset
* Total Records: ~583,000
* Target Variable: `ARR_DEL15`

### Selected Features:

* DAY_OF_MONTH
* DAY_OF_WEEK
* OP_CARRIER_AIRLINE_ID
* ORIGIN_AIRPORT_ID
* DEST_AIRPORT_ID
* DEP_TIME
* ARR_TIME
* DEP_DEL15
* DIVERTED
* DISTANCE


## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib


## 🤖 Machine Learning Model

* Algorithm: **Random Forest Classifier**
* Train-Test Split: 80:20
* Accuracy Achieved: **91.98%**

### Evaluation Metrics:

* Precision
* Recall
* F1-Score
* Confusion Matrix
* Feature Importance Analysis


## 📈 Model Evaluation

The model achieved:

* **Accuracy:** 91.98%
* Strong performance on on-time predictions
* Good balance between precision and recall for delayed flights

### Visualizations Included:

* Feature Importance Graph
* Confusion Matrix Heatmap


## 📂 Project Structure

flight-delay-prediction-ml/
│
├── data/
│   └── flight_data.csv (not included in repo)
│
├── models/
│   └── random_forest_model.pkl (generated after training)
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│
├── main.py
├── .gitignore
└── README.md


## 🚀 How to Run This Project

### 1️⃣ Install Dependencies

pip install -r requirements.txt


### 2️⃣ Train the Model

python main.py

This will:

* Load and preprocess data
* Train the Random Forest model
* Print evaluation metrics
* Generate visualizations
* Save the trained model

### 3️⃣ Run Prediction

python src/predict.py


## 💾 Model Saving

The trained model is saved as:

models/random_forest_model.pkl

This allows the model to be reused for deployment or prediction without retraining.


## 🔍 Key Highlights

✔ Real-world dataset (500K+ records)
✔ Clean modular project structure
✔ Feature selection & preprocessing
✔ Model evaluation with multiple metrics
✔ Confusion matrix visualization
✔ Feature importance analysis
✔ Model persistence using Joblib

## 👩‍💻 Author

**Akshita Kalakonda**


## 📌 Future Improvements

* Hyperparameter tuning using GridSearchCV
* Comparison with Logistic Regression & Gradient Boosting
* Deployment using Streamlit or Flask
* Handling class imbalance using SMOTE
