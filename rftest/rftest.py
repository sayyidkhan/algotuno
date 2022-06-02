import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

#%matplotlib inline

df = pd.read_csv('AAPL.csv')
df.info()
df.columns

dfNoDate=df.drop('Date', axis=1)

np.mean(df[['Difference']])

dfDiff = df[['Difference']]
#dfDiff.hist()

mean = statistics.mean(dfDiff['Difference'])
sd = statistics.stdev(dfDiff['Difference'])

plt.plot(dfDiff['Difference'], norm.pdf(dfDiff['Difference'], mean, sd))
#plt.show()

conditions = [
    (dfDiff['Difference'] <= -3.5),
    (dfDiff['Difference'] < -3.0) & (dfDiff['Difference'] > -3.5),
    (dfDiff['Difference'] < -2.5) & (dfDiff['Difference'] > -3.0),
    (dfDiff['Difference'] < -2.0) & (dfDiff['Difference'] > -2.5),
    (dfDiff['Difference'] < -1.5) & (dfDiff['Difference'] > -2.0),
    (dfDiff['Difference'] <-1.0) & (dfDiff['Difference'] > -1.5),
    (dfDiff['Difference'] <-0.5) & (dfDiff['Difference'] > -1.0),
    (dfDiff['Difference'] <=0) & (dfDiff['Difference'] > -0.5),
    (dfDiff['Difference'] > 0) & (dfDiff['Difference'] < 0.125),
    (dfDiff['Difference'] > 0.125) & (dfDiff['Difference'] < 0.25),
    (dfDiff['Difference'] > 0.25) & (dfDiff['Difference'] < 0.5),
    (dfDiff['Difference'] > 0.5) & (dfDiff['Difference'] < 0.75),
    (dfDiff['Difference'] > 0.75) & (dfDiff['Difference'] < 1.0),
    (dfDiff['Difference'] > 1.0) & (dfDiff['Difference'] < 1.5),
    (dfDiff['Difference'] > 1.5) & (dfDiff['Difference'] < 2.0),
    (dfDiff['Difference'] > 2.0) & (dfDiff['Difference'] < 2.5),
    (dfDiff['Difference'] > 2.5) & (dfDiff['Difference'] < 3.0),
    (dfDiff['Difference'] > 3.0) & (dfDiff['Difference'] < 3.5),
    (dfDiff['Difference'] >= 3.5)
    ]

values = ['n8h', 'n7h', 'n6h', 'n5h', 'n4h', 'n3h','n2h', 'n1h', 'p1qA', 'p1qB', 'p2q','p3q','p4q','p3h', 'p4h','p5h', 'p6h', 'p7h', 'p8h']
df['deviation'] = np.select(conditions, values)
dfNoDate = df.drop('Date', axis=1)

from sklearn.model_selection import train_test_split
X=dfNoDate.drop('deviation', axis=1)
y=dfNoDate['deviation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rfc = RandomForestClassifier(n_estimators=128)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test, rfc_pred))

joblib.dump(rfc, 'model.sav')