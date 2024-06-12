from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Membaca data
data = pd.read_csv("data/nama_file.csv")

# Memilih variabel independen dan dependen
X = data[['konsentrasi_a', 'spl']]
y = data[['panjang_ikan', 'lebar_ikan', 'tinggi_ikan', 'diameter_mata']]

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model regresi
model = LinearRegression()
model.fit(X_train, y_train)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', data=data.to_html())

@app.route('/prediction')
def predictView():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    konsentrasi_a = float(request.form['konsentrasi_a'])
    spl = float(request.form['spl'])
    new_data = np.array([[konsentrasi_a, spl]])
    prediction = model.predict(new_data)
    result = prediction.tolist()[0]

    if any(r < 0 for r in result):
        result = "Kami mendapatkan error dengan nilai morfometrik ikan yang negatif untuk memprediksi SPL dan klorofil-a tersebut, coba value lain yang wajar SPL(22-28) klorofil-a(1-2)"
        return render_template('predict.html', prediction=result, error=None)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    error = {
        'MSE': mse,
    }

    return render_template('predict.html', prediction=result, error=error)


if __name__ == '__main__':
    app.run(debug=True)