from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    try:
        if request.method == "GET":
            result = 0
            hours = request.args.get("hours")
            if isinstance(hours, str):
                hours = int(hours)

                # Machine Learning
                data = pd.read_csv("resources/data.csv")
                X = data.iloc[:, [0]]
                y = data.iloc[:, [1]]
                Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
                model = LinearRegression()
                model.fit(Xtrain, ytrain)
                inp = np.array([hours]).reshape(-1, 1)
                ypred = model.predict(inp)
                ypred = math.floor(ypred[0][0])

                return render_template("success.html", result=ypred, original=hours)
            else:
                return render_template("index.html")

        else:
            return render_template("index.html")
    except:
        return render_template("index.html", message="Something went wrong")


if __name__ == "__main__":
    app.run(debug=True)