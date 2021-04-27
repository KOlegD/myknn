import os
import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

loaded_model = pickle.load(open('knnpickle_file', 'rb'))


@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html', title='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3']),
                           float(request.form['list4'])]])
        pred = int(loaded_model.predict(X_new))
        sort = ['setosa', 'versicolor', 'virginica']
        result = sort[pred]
        return render_template('index.html', title="Это: " + result)


if __name__ == '__main__':
    # app.run(port=8080, host='127.0.0.1')
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
