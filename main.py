from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# ✅ Load model (make sure it was saved with joblib.dump)
diabetes_model = joblib.load("diabetes_model.sav")

@app.get("/")
def root():
    return {"message": "Diabetes Prediction API is running 🚀"}

@app.post("/diabetes_prediction")
def diabetes_pred(input_parameters: ModelInput):
    input_list = [[
        input_parameters.Pregnancies,
        input_parameters.Glucose,
        input_parameters.BloodPressure,
        input_parameters.SkinThickness,
        input_parameters.Insulin,
        input_parameters.BMI,
        input_parameters.DiabetesPedigreeFunction,
        input_parameters.Age
    ]]
    
    prediction = diabetes_model.predict(input_list)
    
    return {
        "prediction": int(prediction[0]),
        "message": "The person is diabetic" if prediction[0] == 1 else "The person is not diabetic"
    }
