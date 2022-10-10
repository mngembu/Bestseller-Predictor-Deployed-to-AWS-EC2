import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('finalModel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    final_features = pd.DataFrame([(features)], columns = ['rating', 'customers_rated', 'price', 'format_Hardcover', 'format_Kindle', 'format_Mass Market Paperback',
                     'format_Paperback', ])
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='The novel is a {}'.format(output))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)