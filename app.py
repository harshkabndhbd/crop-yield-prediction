from flask import Flask, request, render_template
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

model = pickle.load(open('model/model.pkl', 'rb'))
model_columns = pickle.load(open('model/columns.pkl', 'rb'))

df = pd.read_csv('data/yield_df.csv')
countries = sorted(df['Area'].unique())
crops = sorted(df['Item'].unique())

def make_prediction(data):
    input_dict = {col: 0 for col in model_columns}

    input_dict['Year'] = data['year']
    input_dict['average_rain_fall_mm_per_year'] = data['rainfall']
    input_dict['pesticides_tonnes'] = data['pesticides']
    input_dict['avg_temp'] = data['temp']

    input_dict[f"Area_{data['country']}"] = 1
    input_dict[f"Item_{data['crop']}"] = 1

    input_df = pd.DataFrame([input_dict])
    return model.predict(input_df)[0]

@app.route('/')
def home():
    return render_template('index.html', countries=countries, crops=crops)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # -------- MAIN PREDICTION --------
        data = {
            'year': float(request.form['year']),
            'rainfall': float(request.form['rainfall']),
            'pesticides': float(request.form['pesticides']),
            'temp': float(request.form['temperature']),
            'country': request.form['country'],
            'crop': request.form['crop']
        }

        prediction = make_prediction(data)

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

@app.route('/compare', methods=['POST'])
def compare():
    try:
        # Scenario 1
        data1 = {
            'year': float(request.form['year1']),
            'rainfall': float(request.form['rainfall1']),
            'pesticides': float(request.form['pesticides1']),
            'temp': float(request.form['temperature1']),
            'country': request.form['country1'],
            'crop': request.form['crop1']
        }

        # Scenario 2
        data2 = {
            'year': float(request.form['year2']),
            'rainfall': float(request.form['rainfall2']),
            'pesticides': float(request.form['pesticides2']),
            'temp': float(request.form['temperature2']),
            'country': request.form['country2'],
            'crop': request.form['crop2']
        }

        pred1 = make_prediction(data1)
        pred2 = make_prediction(data2)

        # Graph
        plt.figure()
        plt.bar(['Scenario 1', 'Scenario 2'], [pred1, pred2])
        plt.title("Scenario Comparison")

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template('index.html',
                               countries=countries,
                               crops=crops,
                               comparison_text=f'S1: {round(pred1,2)} | S2: {round(pred2,2)}',
                               plot_url=plot_url)

    except Exception as e:
        return render_template('index.html',
                               countries=countries,
                               crops=crops,
                               comparison_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)