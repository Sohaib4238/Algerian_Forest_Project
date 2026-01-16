from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# 1. Initialize App
application = Flask(__name__)
app = application

# 2. Load Models
# Ensure these files exist in the 'models' folder!
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
model = pickle.load(open('models/ridge.pkl', 'rb'))

# 3. Route for Home Page
@app.route("/")
def index():
    return render_template('index.html')

# 4. Route for Prediction
@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Get data from the form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        DC = float(request.form.get('DC'))  # We added this
        ISI = float(request.form.get('ISI'))
        BUI = float(request.form.get('BUI')) # We added this
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # Combine into a list (Order must match your training data!)
        # Order: Temp, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, Classes, Region
        new_data = [[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, Classes, Region]]
        
        # Scale the data
        new_data_scaled = scaler.transform(new_data)
        
        # Predict
        result = model.predict(new_data_scaled)

        # Show result on the page
        return render_template('home.html', result=result[0])

    else:
        # If the user just goes to /predictdata directly, show the empty form
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")