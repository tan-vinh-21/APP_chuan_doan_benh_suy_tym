from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('D:\\School\\APP_chuan_doan_benh_suy_tym\\model_1.pkl', 'rb'))
sc_a = pickle.load(open('D:\\School\\APP_chuan_doan_benh_suy_tym\\scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        A = [float(x) for x in request.form.values()]
        if len(A) != 12: 
            return render_template('index.html', result='Input features mismatch')
        A = np.array(A).reshape(1, -1)
        A = sc_a.transform(A)
        
        model_prediction = model.predict(A)
        
        prediction = "Prediction: %d" % model_prediction[0]  # 0 là không, 1 là có
        
        return render_template('index.html', result=prediction)
    except Exception as e:
        return render_template('index.html', result=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
