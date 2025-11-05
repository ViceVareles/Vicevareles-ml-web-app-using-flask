# 🩺 App de Predicción de Diabetes con Flask

https://vicevareles-ml-web-app-using-flask.onrender.com

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

## 🧪 Cómo probar la aplicación paso a paso

1. **Arranca el servidor** con `python app.py`; verás en la terminal un mensaje similar a `Running on http://127.0.0.1:5000/`.
2. **Abre el navegador** y navega a `http://127.0.0.1:5000/` (si estás en un entorno remoto, utiliza el reenvío de puertos correspondiente).
3. **Introduce valores de ejemplo** en el formulario y pulsa **Predict** para ver la respuesta del modelo.
4. **Prueba validaciones**: si dejas un campo vacío o introduces un texto no numérico, la página mostrará un mensaje de error sin detener el servidor.
5. **Detén la aplicación** con `Ctrl+C` en la terminal cuando termines.

Si prefieres ejecutar una comprobación rápida sin abrir el navegador, puedes compilar los archivos con:

```bash
python -m compileall app.py templates/index.html
```

Ese comando verifica que no existan errores de sintaxis en el código Python ni en la plantilla principal.

> **Importante:** El modelo se distribuye incrustado en el código, sin dependencias externas como scikit-learn. Los coeficientes del `StandardScaler` y del `LassoCV` se reimplementaron en una clase ligera de Python, de modo que no necesitas archivos binarios ni bibliotecas científicas para ejecutar las predicciones.

El predictor espera exactamente los diez campos clínicos que muestra el formulario (edad, sexo, IMC, presión arterial y los seis indicadores bioquímicos). Asegúrate de que cada uno sea numérico para obtener una predicción válida.



   ```bash
   python app.py
   ```

5. Abre tu navegador en `http://127.0.0.1:5000` para comenzar a realizar predicciones.

> **Importante:** El archivo `lasso_cv_diabetes_model.pkl` debe permanecer en la raíz del proyecto para que la aplicación pueda
> cargar el modelo entrenado.


