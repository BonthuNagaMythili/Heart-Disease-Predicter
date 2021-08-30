from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__, template_folder='templates')

filename = 'saved_model.pkl'
loaded_model = pickle.load( open( filename, 'rb' ) )

@app.route('/')
def home():
    return render_template( 'input.html' )


@app.route('/output', methods=["POST", "GET"] )
def output():
    if request.method == "POST":
        age = request.form["age"]
        gender = request.form["gender"]
        cp = request.form["cp"]
        rbp = request.form["rbp"]
        chol = request.form['chol']
        sug = request.form['sugar']
        ecg = request.form['ecg']
        hr = request.form['hr']
        anigma = request.form['exang']
        peak = request.form['op']
        slope = request.form['slope']
        vess = request.form['ca']
        thal = request.form['thal']

    inp = [[int( age ), int( gender ), int( cp ), int( rbp ), int( chol ), int( sug ), int( ecg ), int( hr ),
            int( anigma ), float( peak ),
            int( slope ), int( vess ), int( thal )]]

    arr = np.asarray( inp )
    arr = arr.reshape( 1, -1 )

    pred = loaded_model.predict(arr)

    if pred:
        return render_template('input.html', prediction_text='You must consult a Cardiologist!')
    else:
        return render_template('input.html', prediction_text='You Dont have any Heart disease!')

if __name__ == "__main__":
    app.run()