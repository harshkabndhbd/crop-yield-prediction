from flask import Flask, request, render_template
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load model
model = pickle.load(open('model/model.pkl', 'rb'))
model_columns = pickle.load(open('model/columns.pkl', 'rb'))

# Load dataset
df = pd.read_csv('data/yield_df.csv')
countries = sorted(df['Area'].unique())
crops = sorted(df['Item'].unique())

@app.route('/')
def home():
    return render_template('index.html', countries=countries, crops=crops)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        year = float(request.form['year'])
        rainfall = float(request.form['rainfall'])
        pesticides = float(request.form['pesticides'])
        temp = float(request.form['temperature'])
        country = request.form['country']
        crop = request.form['crop']

        input_dict = {col: 0 for col in model_columns}

        input_dict['Year'] = year
        input_dict['average_rain_fall_mm_per_year'] = rainfall
        input_dict['pesticides_tonnes'] = pesticides
        input_dict['avg_temp'] = temp

        input_dict[f'Area_{country}'] = 1
        input_dict[f'Item_{crop}'] = 1

        input_df = pd.DataFrame([input_dict])

        prediction = model.predict(input_df)[0]

        # Graph (Predicted vs Average)
        avg_yield = df['hg/ha_yield'].mean()

        plt.figure()
        plt.bar(['Predicted', 'Average'], [prediction, avg_yield])
        plt.title("Yield Comparison")

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template('index.html',
                               countries=countries,
                               crops=crops,
                               prediction_text=f'Predicted Yield: {round(prediction,2)}',
                               plot_url=plot_url)

    except Exception as e:
        return render_template('index.html',
                               countries=countries,
                               crops=crops,
                               prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)