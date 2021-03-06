import pandas as pd
import pickle
from flask import Flask, request
from cardio import Cardio

app = Flask(__name__)

# load model
model = pickle.load(open('model/xgbc.pkl', 'rb'))

@app.route("/predict", methods=['POST'])
def predict():
    test_json = request.get_json()

    # collect data
    if test_json:
        if isinstance(test_json, dict):
            df_raw = pd.DataFrame(test_json, index=[0])
        else:
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

    # instanciate data preparation
    pipeline = Cardio()
    df1 = pipeline.data_preparation(df_raw)

    # prediction
    pred = model.predict(df1)
    df_raw['prediction'] = pred

    return df_raw.to_json(orient='records')

if __name__ == '__main__':
    # start flask
    app.run(host='0.0.0.0', port='5000')