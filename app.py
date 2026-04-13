from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# =========================
# LOAD MODEL + DATA
# =========================

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load model
model_path = os.path.join(base_dir, 'model', 'model.pkl')
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Load dataset
data_path = os.path.join(base_dir, 'data', 'yield_df.csv')
df = pd.read_csv(data_path)

countries = sorted(df['Area'].unique())
crops = sorted(df['Item'].unique())

# =========================
# HOME
# =========================

@app.route('/')
def home():
    return render_template('index.html', countries=countries, crops=crops)

# =========================
# PREDICT
# =========================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        country = request.form.get('country')
        crop = request.form.get('crop')
        rainfall = float(request.form.get('rainfall'))
        pesticide = float(request.form.get('pesticide'))
        temperature = float(request.form.get('temperature'))

        input_data = np.array([[rainfall, pesticide, temperature]])
        prediction = model.predict(input_data)[0]

        # GRAPH
        plt.figure()
        plt.bar(['Yield'], [prediction])
        plt.title('Predicted Yield')

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

        return render_template(
            'index.html',
            prediction=round(prediction, 2),
            countries=countries,
            crops=crops,
            graph_url=graph_url
        )

    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# RUN (RENDER FIX)
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
