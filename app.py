from flask import Flask, render_template, request
import numpy as np
import joblib

# Load the trained model
model = joblib.load('model_rf_10features.pkl')

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Collect input data from form
        team = int(request.form['team'])
        targeted_productivity = float(request.form['targeted_productivity'])
        smv = float(request.form['smv'])
        wip = float(request.form['wip'])
        over_time = int(request.form['over_time'])
        incentive = int(request.form['incentive'])
        idle_time = float(request.form['idle_time'])
        idle_men = int(request.form['idle_men'])
        no_of_style_change = int(request.form['no_of_style_change'])
        no_of_workers = float(request.form['no_of_workers'])

        # Prepare data for model prediction
        final_input = np.array([[team, targeted_productivity, smv, wip,
                                 over_time, incentive, idle_time, idle_men,
                                 no_of_style_change, no_of_workers]])

        # Predict the productivity
        prediction = round(model.predict(final_input)[0], 2)

        # Determine productivity level
        if prediction >= 0.75:
            level = "High Productive"
        elif prediction >= 0.5:
            level = "Medium Productive"
        else:
            level = "Low Productive"

        # Render the result page with prediction and level
        return render_template('submit.html', prediction=prediction, level=level)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)