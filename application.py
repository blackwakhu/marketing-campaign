from crypt import methods
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('Randomforest.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    education_list= ['2n cycle', 'basic','graduation','master','phd'] 
    initial = [float(x) for x in request.form.values()]
    final = np.array(initial).reshape(1,-1)
    response = model.predict(final)
    if response == '0':
        prediction = 'negative'
    else:
        prediction = 'positive'
    return render_template('index.html', prediction=prediction)

if __name__ =="__main__":
    app()