# Customer Churn Prediction API

## Project Overview
This project predicts whether a customer will churn using a machine learning model.

## Steps
1. Data preprocessing using pandas
2. Feature encoding using get_dummies
3. Model training using RandomForest
4. Model saved as model.pkl
5. FastAPI used to create API

## API Endpoint
POST /predict

## Example Input
```json
{
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 800,
  "gender_Male": 1
}