import pandas as pd
import joblib
import json

df_pred = pd.read_csv('AAPL_pred.csv')
loaded_model = joblib.load('model.sav')
result = loaded_model.predict(df_pred)
print(type(result))

output = result[0]
print(type(output))

jobj = {
'message':output
}

print(type(jobj))
print (jobj)
