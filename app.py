from flask import Flask, render_template, request
import numpy as np
import joblib
import os


app = Flask(__name__)


model = joblib.load("iris_lr.pkl")
scaler = joblib.load("iris_minmax_scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
       
        sepal_length = float(request.form["sepal_length"])
        sepal_width  = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width  = float(request.form["petal_width"])

       
        features = np.array([[sepal_length,
                              sepal_width,
                              petal_length,
                              petal_width]])

       
        features_scaled = scaler.transform(features)

      
        pred = model.predict(features_scaled)[0]   
        prediction = str(pred)

   
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host="0.0.0.0", port=port, debug=True)
