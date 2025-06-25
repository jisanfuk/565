
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

def get_streak(row):
    s = 1
    if row['prev1'] == row['prev2'] == row['prev3']:
        s = 3
    elif row['prev2'] == row['prev3']:
        s = 2
    return s

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        prev1 = request.form['prev1'].upper()
        prev2 = request.form['prev2'].upper()
        prev3 = request.form['prev3'].upper()
        data = pd.DataFrame([{'prev1': prev1, 'prev2': prev2, 'prev3': prev3}])
        data['streak_len'] = data.apply(get_streak, axis=1)
        data = pd.get_dummies(data)
        for col in model.feature_names_in_:
            if col not in data.columns:
                data[col] = 0
        data = data[model.feature_names_in_]
        pred = model.predict(data)[0]
        prediction = 'BIG' if pred == 1 else 'SMALL'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
