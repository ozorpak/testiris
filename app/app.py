# # app.py
# from flask import Flask, request
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load model
# with open('model.pkl', 'rb') as f:
#     model = pickle.load(f)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     prediction = model.predict(np.array(data['input']).reshape(1, -1))
#     return {'prediction': int(prediction[0])}

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

    # app.py
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = request.form.get('input')
        data = [float(i) for i in data.split(',')]
        prediction = model.predict(np.array(data).reshape(1, -1))
        return {'prediction': int(prediction[0])}
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)