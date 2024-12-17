from flask import Flask, request, render_template
app = Flask(__name__)
@app.route('/')
def hello():
    return render_template('abc.html')#'Hello, World!'

@app.route('/predict', methods = ['POST']) 
def predict():
     A = [float(x) for x in request.form.values()]
     print(A)
     kq="Toi dang thu ket qua: " + str(A)
     return render_template('abc.html', result = kq)
if __name__== "__main__":
    app.run(debug=True)