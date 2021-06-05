# Import Libraries
import numpy as np
from flask import Flask, jsonify, request, render_template
import joblib
import statsmodels.api as sm
import numpy
from sklearn.preprocessing import MinMaxScaler

# Object initialization
app = Flask(__name__)

# Load Model Files
model_load = joblib.load('D:\Data Science\Python Projects\PyCharm\ML Model Deployment\BOOM Bike Sharing Regression Model\models\Boom_Bike_lrmodel.pkl')
scaler_object = joblib.load('D:\Data Science\Python Projects\PyCharm\ML Model Deployment\BOOM Bike Sharing Regression Model\models\Boom_Bike_scaler.pkl')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        init_features = [float(x) for x in request.form.values()]
        init_features=[1.0]+init_features
        final_features = np.array(init_features).reshape(1,12)
        final_features[0][[3,11,4]] = scaler_object.transform(final_features[0][[3,11,4]].reshape(1,3))
        final_features=final_features[0][:11]
        final_features=final_features.reshape(1,11)
        output = int(model_load.predict(final_features).tolist()[0])
        return render_template('index.html',predicted_count = 'Sales Count: {} '.format(output))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)