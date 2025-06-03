import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))  # Load your trained model

@app.route('/')
def home():
    return render_template('index.html')  # Form HTML page

@app.route('/predict', methods=['POST'])
def predict():
    # Get inputs from form and convert to float
    input_features = [
        float(request.form['bedroom']),
        float(request.form['bathroom']),
        float(request.form['living_area']),
        float(request.form['floors']),
        float(request.form['year'])
    ]

    # Predict using the model
    features_value = np.array([input_features])
    predicted_price = model.predict(features_value)[0]

    # Return result to the same page
    return render_template(
        'index.html',
        prediction="Predicted House Price: ${:,.2f}".format(predicted_price)
    )

if __name__ == '__main__':
    app.run(debug=True)
