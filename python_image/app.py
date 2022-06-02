import json
import pandas as pd
import joblib

def lambda_handler(event, context):
    df_pred = pd.read_csv('AAPL_pred.csv')
    loaded_model = joblib.load('model.sav')
    result = loaded_model.predict(df_pred)
    output = result[0]
    return {
        'message' : output
    }

'''
output must be able to convert into json format
'''
