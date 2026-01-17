# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
# Ensure the path matches your structure. If app.py is in root, model is in model/
try:
    model = joblib.load('model/house_price_model.pkl')
except FileNotFoundError:
    print("Error: Model file not found. Please run Part A to generate the .pkl file.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get data from form
            overall_qual = int(request.form['OverallQual'])
            gr_liv_area = float(request.form['GrLivArea'])
            total_bsmt_sf = float(request.form['TotalBsmtSF'])
            garage_cars = int(request.form['GarageCars'])
            full_bath = int(request.form['FullBath'])
            year_built = int(request.form['YearBuilt'])

            # Create numpy array for prediction
            # The order must match the training data columns exactly
            features = np.array([[overall_qual, gr_liv_area, total_bsmt_sf, garage_cars, full_bath, year_built]])
            
            # Predict
            prediction = model.predict(features)
            output = round(prediction[0], 2)

            return render_template('index.html', prediction_text=f'Estimated House Price: ${output:,.2f}')
        
        except Exception as e:
            return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)