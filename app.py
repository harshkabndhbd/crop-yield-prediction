from flask import Flask, render_template, request
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        rainfall = request.form.get('rainfall')
        pesticide = request.form.get('pesticide')
        temperature = request.form.get('temperature')

        # Check empty
        if not rainfall or not pesticide or not temperature:
            return render_template('index.html', error="Please fill all fields")

        # Convert
        rainfall = float(rainfall)
        pesticide = float(pesticide)
        temperature = float(temperature)

        # Simple prediction logic
        prediction = (rainfall * 2) + (pesticide * 3) + (temperature * 5)

        # Graph
        plt.figure()
        plt.bar(['Yield'], [prediction])
        plt.title("Predicted Yield")

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

        return render_template(
            'index.html',
            prediction=round(prediction, 2),
            graph_url=graph_url
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
