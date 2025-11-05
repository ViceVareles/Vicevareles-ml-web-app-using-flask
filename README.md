# 🩺 App de Predicción de Diabetes con Flask

Esta aplicación web utiliza un modelo de regresión LassoCV entrenado previamente para estimar la progresión de la diabetes a
partir de variables clínicas. La interfaz está construida con Flask y Bootstrap 5, y permite ingresar los valores clínicos de un
paciente para obtener una predicción inmediata.

## 🧰 Requisitos

- Python 3.11 o superior
- pip para instalar dependencias

## 🚀 Puesta en marcha

1. Clona el repositorio y entra al directorio del proyecto.
2. (Opcional) Crea un entorno virtual.
3. Instala las dependencias necesarias:

   ```bash
   pip install -r requirements.txt
   ```

4. Inicia la aplicación:

   ```bash
   python app.py
   ```

5. Abre tu navegador en `http://127.0.0.1:5000` para comenzar a realizar predicciones.

> **Importante:** El archivo `lasso_cv_diabetes_model.pkl` debe permanecer en la raíz del proyecto para que la aplicación pueda
> cargar el modelo entrenado.


