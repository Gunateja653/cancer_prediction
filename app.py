from flask import Flask, request, render_template, redirect, url_for
import pickle
import numpy as np


app = Flask(__name__)


model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   
    input_features = [float(x) for x in request.form.values()]
    final_features = [np.array(input_features)]
    
    prediction = model.predict(final_features)
    

    output = prediction[0]

    if output == 1:
        result = "The diagnosis is malignant (cancerous), SO indicating the presence of cancer --__--"
    else:
        result = "The diagnosis is benign (non-cancerous), SO indicating the absence of cancer --__--"
    
    return redirect(url_for('result', result=result))

@app.route('/result/<result>')
def result(result):
    return render_template('result.html', prediction_text=f'Diagnosis: {result}')

if __name__ == "__main__":
    app.run(debug=True)
