import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model from the pickle file
with open('iris_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Define a route to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')


# Define a route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get feature values from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        print(f"Exception: {e}")
        return render_template('index.html', error_message=str(e))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)