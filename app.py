from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)


file = open("./gradient_boosting_regressor_model.pkl", 'rb')
model = pickle.load(file)

data = pd.read_csv('./clean_data.csv')
data.head()

@app.route('/')
def index():
    sex = sorted(data['sex'].unique())
    smoker = sorted(data['smoker'].unique())
    region = sorted(data['region'].unique())
    return render_template('index.html', sex= sex, smoker= smoker, region= region)

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    sex = request.form.get('sex')
    bmi = float(request.form.get('bmi'))
    children = int(request.form.get('children'))
    smoker = request.form.get('smoker')
    region = request.form.get('region')

    prediction = model.predict(pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']))

    return str(prediction[0])           

if __name__=="__main__":
    app.run(debug=True)
from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Define base path (go up one level from 'app' folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get 'app' folder path
MODEL_DIR = os.path.join(BASE_DIR, "..")  # Go up one level to 'model' folder

# Load Model
model_path = os.path.join(MODEL_DIR, "gradient_boosting_regressor_model.pkl")
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load Data for Dropdown Values
data_path = os.path.join(MODEL_DIR, "clean_data.csv")
data = pd.read_csv(data_path)

@app.route('/')
def index():
    sex = sorted(data['sex'].unique())
    smoker = sorted(data['smoker'].unique())
    region = sorted(data['region'].unique())
    return render_template('index.html', sex=sex, smoker=smoker, region=region)

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get('age'))
    sex = request.form.get('sex')
    bmi = float(request.form.get('bmi'))
    children = int(request.form.get('children'))
    smoker = request.form.get('smoker')
    region = request.form.get('region')

    # Encoding categorical variables (ensure these match your model's training data)
    sex_map = {'male': 0, 'female': 1}
    smoker_map = {'no': 0, 'yes': 1}
    region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}

    sex = sex_map.get(sex, 0)
    smoker = smoker_map.get(smoker, 0)
    region = region_map.get(region, 0)

    # Making prediction
    prediction = model.predict(pd.DataFrame([[age, sex, bmi, children, smoker, region]], 
                columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region']))

    return str(prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
