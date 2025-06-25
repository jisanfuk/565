import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

data = [
    ['BIG','BIG','BIG','SMALL'], ['BIG','BIG','SMALL','BIG'], ['SMALL','BIG','BIG','BIG'],
    ['BIG','SMALL','BIG','SMALL'], ['SMALL','SMALL','SMALL','BIG'], ['BIG','BIG','BIG','SMALL'],
    ['SMALL','BIG','SMALL','SMALL'], ['BIG','BIG','SMALL','BIG'], ['BIG','SMALL','SMALL','SMALL'],
    ['SMALL','BIG','BIG','SMALL'], ['BIG','BIG','BIG','BIG'], ['SMALL','SMALL','SMALL','SMALL']
]
df = pd.DataFrame(data, columns=['prev1','prev2','prev3','next'])
df['streak_len'] = df.apply(lambda r: 3 if r['prev1']==r['prev2']==r['prev3'] else (2 if r['prev2']==r['prev3'] else 1), axis=1)
X = pd.get_dummies(df[['prev1','prev2','prev3']])
X['streak_len'] = df['streak_len']
y = df['next'].map({'BIG':1, 'SMALL':0})
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)
joblib.dump(model, 'model.pkl')
print("✅ Model created successfully as model.pkl")
