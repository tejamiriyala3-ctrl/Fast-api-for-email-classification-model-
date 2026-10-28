import pandas as pd 
import numpy as np 
import joblib 
from fastapi import FastAPI
from pydantic import BaseModel
model = joblib.load('spam_ham_model.pkl')
vectorizer = joblib.load('vectorizer1.pkl')
app = FastAPI()

#input data 
class Email(BaseModel):
    message:str

@app.get("/")
def home():
    return{"message": "email classifier api is running"}

#prediction route 
@app.post("/predict")
def predict_email(email:Email):
    #convert text into numbers
    data = vectorizer.transform([email.message])

    #make prediction 
    prediction = model.predict(data)

    return{
        "emial":email.message,
        "prediction":prediction[0]
    }
