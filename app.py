from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from docx import Document

app = Flask(__name__)
app.secret_key = 'your_secret_key'  

model_path = 'model/model.pkl'
model, X_train_columns, model_performance = joblib.load(model_path)
predictions_log_path = 'model/predictions_log.csv'

def encode_categorical(input_data):
    categorical_cols = {
        'sex': {'male': 1, 'female': 0},
        'race': {'white': 0, 'black': 1, 'asian': 2, 'other': 3},
        'ethnicity': {'hispanic': 1, 'non-hispanic': 0},
        'facility': {'Community Cancer Program': 0, 'Comprehensive Community Cancer Program': 1, 'Academic/Research Program': 2},
        'grade': {'Low grade': 0, 'Intermediate grade': 1, 'High grade': 2},
        'charlson_deyo_score': {'0': 0, '1': 1, '2': 2, '3': 3},
        'insurance_status': {'Insured': 0, 'Uninsured': 1},
        'laterality': {'Right': 0, 'Left': 1, 'Bilateral': 2},
        'lymph_vascular_invasion': {'Yes': 1, 'No': 0},
        'surgery': {'Yes': 1, 'No': 0},
        'radiotherapy': {'Yes': 1, 'No': 0},
        'puf_vital_status': {'Alive': 1, 'Deceased': 0}
    }
    
    for col, mapping in categorical_cols.items():
        if col in input_data.columns:
            input_data[col] = input_data[col].map(mapping)
    
    return input_data

def get_os_description(os_months):
    if os_months < 12:
        return "Poor"
    elif 12 <= os_months < 24:
        return "Fair"
    elif 24 <= os_months < 36:
        return "Good"
    else:
        return "Excellent"

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form.to_dict()
            input_data = pd.DataFrame([data])
            
            input_data.columns = input_data.columns.str.lower()
            
            input_data = encode_categorical(input_data)
            
            numeric_cols = ['age', 'income', 'regional_nodes_positive', 'regional_nodes_examined']
            for col in numeric_cols:
                if col in input_data.columns:
                    input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
            
            input_data = input_data.fillna(input_data.mean())
            
            missing_cols = set(X_train_columns) - set(input_data.columns)
            for col in missing_cols:
                input_data[col] = 0
            input_data = input_data[X_train_columns]

            if input_data.isnull().values.any():
                raise ValueError("Invalid input: Please ensure all fields are filled correctly.")

            prediction = model.predict(input_data)
            
            os_description = get_os_description(prediction[0])

            if not os.path.exists(predictions_log_path):
                input_data.to_csv(predictions_log_path, index=False)
            else:
                input_data.to_csv(predictions_log_path, mode='a', header=False, index=False)
            
           
            plt.figure(figsize=(10, 6))
            plt.bar(['Predicted OS'], [prediction[0]], color='#007bff')
            plt.title("Predicted Overall Survival (OS)")
            plt.ylabel("Months")
            plt.ylim(0, max(60, prediction[0] * 1.2))
            
            
            img = BytesIO()
            plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            
            return render_template('result.html', 
                                   prediction=round(prediction[0], 2), 
                                   mse=round(model_performance['mse'], 4), 
                                   mae=round(model_performance['mae'], 4), 
                                   r2=round(model_performance['r2'], 4), 
                                   os_description=os_description, 
                                   plot_url=plot_url)
        except Exception as e:
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

@app.route('/export')
def export():
    if os.path.exists(predictions_log_path):
        df = pd.read_csv(predictions_log_path)
        document = Document()
        document.add_heading('RCC Survival Prediction Log', 0)

        table = document.add_table(df.shape[0] + 1, df.shape[1])
        for j in range(df.shape[-1]):
            table.cell(0, j).text = df.columns[j]

        for i in range(df.shape[0]):
            for j in range(df.shape[-1]):
                table.cell(i + 1, j).text = str(df.values[i, j])

        export_path = os.path.join('model', 'rcc_prediction_log.docx')
        document.save(export_path)
        return redirect(url_for('download_file', filename='rcc_prediction_log.docx'))
    return "No predictions to export."

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory('model', filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)