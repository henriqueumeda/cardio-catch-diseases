import pandas as pd
import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from cardio import Cardio

app = FastAPI()

# load model
model = pickle.load(open('model/xgbc.pkl', 'rb'))

class Patient(BaseModel):
    weight: float
    ap_hi: int
    ap_lo: int
    age_year: int
    bmi: float
    cholesterol: int
    gluc: int
    hypertension: int

@app.post('/predict')
def predict(patient: Patient):

    # collect data
    if patient:
        df_raw = pd.DataFrame(dict(patient).values(), index=dict(patient).keys()).T

    # instanciate data preparation
    pipeline = Cardio()
    df1 = pipeline.data_preparation(df_raw)

    # prediction
    pred = model.predict(df1)
    df_raw['prediction'] = pred

    return df_raw.to_json(orient='records')

if __name__ == '__main__':
    # start fastapi
    uvicorn.run('app:app', host='127.0.0.1', port=8000)
