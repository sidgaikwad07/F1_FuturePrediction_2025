# F1_FuturePrediction_2025

# 🏁 Monaco GP 2025 Predictions using FP1/FP2 + Machine Learning

## This project uses only FP1 and FP2 session data to predict the outcome of the Monaco Grand Prix 2025, skipping qualifying entirely to simulate a real-world data-constrained prediction scenario.
## Built using FastF1 and XGBoost, the pipeline models race finishing positions using practice telemetry, weather, tyre strategies, and Monaco-specific performance modifiers.

🚀 Project Highlights

📊 Machine Learning Model: XGBoost regression model trained exclusively on 2025 race outcomes
📡 Telemetry Powered: Pulls FP1/FP2 lap data and weather conditions using FastF1
🌧️ Weather + Track Awareness: Models impact of rainfall, track temperature, and humidity
🧠 Monaco-Aware Feature Stack: Rolling lap averages, rookie flags, teammate deltas, tyre compound encoding
🏎️ No Qualifying Data Used: Trained and predicted without using Q1/Q2/Q3
📈 Model Evaluation: Achieved MAE = 0.76 and R² = 0.92 on validation
📊 Visualization: Includes model comparison plot across 5 algorithms

## Project Structure
F1_Monaco_FPOnly_2025/
├── data/                          # Cleaned FP1/FP2 data + weather
│   └── monaco_2025_fp_cleaned.csv
├── clean_data/
│   └── monaco_2025_fp_engineered.csv
├── models_2025_only/
│   └── XGBoost_2025.pkl           # Trained model (not committed)
├── assets/
│   └── mae_race_quali_cleaned.png
├── feature_engineering/
│   └── feature_engineer_monaco_fp.py
├── modelling/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
├── predictions/
│   └── predict_monaco2025_fp_only.py
├── monaco_2025_prediction/
│   └── top10_monaco_2025_predictions.csv
└── README.md

⚙️ How It Works
1. 🛠 Feature Engineering
	•	Loads FP1 + FP2 lap data for Monaco 2025
	•	Computes rolling 3-lap average pace per driver
	•	Flags rookies and models teammate deltas
	•	One-hot encodes tyre compounds and weather conditions
	•	Outputs enriched dataset for prediction

2. 🧪 Model Training
	•	Trains an XGBoostRegressor on 2025 GPs only (race position as target)
	•	Evaluates MAE, RMSE, and R²
	•	Compares with SVR, LinearRegression, LightGBM, and RandomForest

3. 📉 Model Evaluation
   ## 📉 Model Comparison (Race + Quali Cleaned)

We evaluated five different regression models on cleaned 2025 race and qualifying data. Mean Absolute Error (MAE) was used as the primary metric to assess prediction accuracy.

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1hQe9yz4soc9sUHK20ceSn4VXGm1W6Q6y" width="600"/>
</p>

> XGBoost achieved the **lowest MAE**, followed by LightGBM and RandomForest, making it the most effective choice for this context.
4. 🏁 Monaco GP 2025 Prediction
	•	Uses the engineered FP1/FP2 data
	•	Aligns features with the trained model
	•	Predicts race position per lap and selects best lap per driver
	•	Outputs a Top 10 classification table

Why XGBoost?
XGBoost was chosen based on its:
	•	📈 Consistently low MAE
	•	⚡ Ability to handle missing or sparse practice data
	•	🌲 Superior handling of non-linear interactions
	•	🧰 Robust tuning and interpretability features

🏆 Top 3 Predicted Finishers – Monaco 2025
	1.	Max Verstappen – Red Bull Racing
	2.	Lando Norris – McLaren
	3.	Yuki Tsunoda – Red Bull Racing

🙌 Acknowledgements
Special thanks to:
	•	FastF1 for the amazing telemetry library
	•	XGBoost for its performance and reliability
	•	The F1 Data Science Community for all the inspiration
	•	And of course… Monaco, for always keeping us guessing 
