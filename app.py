from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Cargar el modelo entrenado
with open("lasso_cv_diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Extraer valores del formulario
            features = [
                float(request.form["age"]),
                float(request.form["sex"]),
                float(request.form["bmi"]),
                float(request.form["bp"]),
                float(request.form["s1"]),
                float(request.form["s2"]),
                float(request.form["s3"]),
                float(request.form["s4"]),
                float(request.form["s5"]),
                float(request.form["s6"])
            ]
            # Predecir con el modelo
            prediction = model.predict([features])[0]
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)