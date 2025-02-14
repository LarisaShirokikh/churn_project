from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
import json

# Загружаем модель и Scaler
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# Оптимальный порог
THRESHOLD = 0.30

# Правильные признаки (из модели)
MODEL_FEATURES = model.feature_names_in_

# Описание входных данных (валидатор)
class CustomerData(BaseModel):
    SeniorCitizen: int
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender_Male: int
    Partner_Yes: int
    Dependents_Yes: int
    PhoneService_Yes: int
    MultipleLines_No_phone_service: int  # Исправлено
    MultipleLines_Yes: int
    InternetService_Fiber_optic: int  # Исправлено
    InternetService_No: int
    OnlineSecurity_No_internet_service: int  # Исправлено
    OnlineSecurity_Yes: int
    OnlineBackup_No_internet_service: int  # Исправлено
    OnlineBackup_Yes: int
    DeviceProtection_No_internet_service: int  # Исправлено
    DeviceProtection_Yes: int
    TechSupport_No_internet_service: int  # Исправлено
    TechSupport_Yes: int
    StreamingTV_No_internet_service: int  # Исправлено
    StreamingTV_Yes: int
    StreamingMovies_No_internet_service: int  # Исправлено
    StreamingMovies_Yes: int
    Contract_One_year: int  # Исправлено
    Contract_Two_year: int  # Исправлено
    PaperlessBilling_Yes: int
    PaymentMethod_Credit_card_automatic: int  # Исправлено
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int
# Главная страница API
app = FastAPI(title="Churn Prediction API")


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running!"}

print("✅ Признаки, которые ждёт модель:")
print(MODEL_FEATURES)

@app.post("/predict/")
def predict_churn(data: CustomerData):
    df = pd.DataFrame([data.model_dump()])

    # ✅ Добавляем вычисляемые признаки
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["ContractType"] = df[['Contract_One_year', 'Contract_Two_year']].idxmax(axis=1).map({
    'Contract_One_year': 1,
    'Contract_Two_year': 2
}).fillna(0).astype(int)
    df["HighSpender"] = (df["MonthlyCharges"] > 70).astype(int)

    # Масштабируем числовые признаки
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

    # ✅ Убеждаемся, что порядок признаков соответствует обучению модели
    df = df[MODEL_FEATURES]

    # Делаем предсказание вероятности
    probability = model.predict_proba(df)[:, 1][0]

    # Определяем класс по порогу
    churn_prediction = 1 if probability >= THRESHOLD else 0

    return {
        "churn_probability": round(probability, 4),
        "churn_prediction": churn_prediction
    }