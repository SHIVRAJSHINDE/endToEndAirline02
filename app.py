from flask import Flask, request, render_template
from flask_cors import cross_origin
import pandas as pd
from src.predictionPipeline import CustomData

app = Flask(__name__)


@app.route("/")
#@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
#@cross_origin()
def predict():
    if request.method=="GET":
        return render_template(home.html)
    else:
        custom_Data = CustomData()
        output = custom_Data.receiveDataFromWeb(
                          Airline = request.form.get('Airline'),
                          Date_of_Journey = request.form.get('Dep_Time'),
                          Source = request.form.get('Source'),
                          Destination = request.form.get('Destination'),
                          Dep_Time = request.form.get('Dep_Time'),
                          Arrival_Time = request.form.get('Arrival_Time'),
                          Duration = request.form.get('Duration'),
                          Total_Stops = request.form.get('Total_Stops')
        
                    )


        return render_template('home.html', prediction_text="Your Flight price is Rs. {}".format(output))

    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
