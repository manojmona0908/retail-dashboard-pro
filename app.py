from flask import Flask, render_template, request
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# =========================================================
# LOAD MODELS
# =========================================================

# Promotion
with open("sales_model.pkl", "rb") as f:
    sales_model = pickle.load(f)

with open("quantity_model.pkl", "rb") as f:
    quantity_model = pickle.load(f)

# Forecast
with open("forecast_model.pkl", "rb") as f:
    forecast_model = pickle.load(f)

# Churn
with open("churn_model.pkl", "rb") as f:
    churn_model = pickle.load(f)

with open("churn_scaler.pkl", "rb") as f:
    churn_scaler = pickle.load(f)

# NLP
with open("nlp_model.pkl", "rb") as f:
    nlp_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# =========================================================
# HOME
# =========================================================
@app.route("/")
def home():
    return render_template("home.html")


# =========================================================
# DASHBOARDS
# =========================================================
@app.route("/retail_dashboard")
def retail_dashboard():
    return render_template("retail_dashboard.html")


@app.route("/customer_dashboard")
def customer_dashboard():
    return render_template("customer_dashboard.html")


# =========================================================
# CHURN
# =========================================================
@app.route("/churn")
def churn():
    return render_template("churn.html", churn_prob=None)


@app.route("/predict_churn", methods=["POST"])
def predict_churn():

    recency = float(request.form["recency"])
    frequency = float(request.form["frequency"])
    monetary = float(request.form["monetary"])

    input_data = np.array([[recency, frequency, monetary]])
    input_scaled = churn_scaler.transform(input_data)

    prediction = churn_model.predict(input_scaled)[0]
    probability = churn_model.predict_proba(input_scaled)[0][1]

    churn_percentage = round(probability * 100, 2)

    return render_template(
        "churn.html",
        churn_prob=churn_percentage
    )


# =========================================================
# PROMOTION
# =========================================================
@app.route("/promotion")
def promotion():
    return render_template("promotion.html")


@app.route("/predict_promotion", methods=["POST"])
def predict_promotion():

    promo_flag = float(request.form["promo"])
    discount = float(request.form["discount"])

    # Base scenario (No promotion, no discount)
    base_input = np.array([[1, 0, 0]])
    base_sales = sales_model.predict(base_input)[0]
    base_quantity = quantity_model.predict(base_input)[0]
    base_revenue = base_sales * base_quantity

    # Current scenario
    input_data = np.array([[1, promo_flag, discount]])

    predicted_sales = sales_model.predict(input_data)[0]
    predicted_quantity = quantity_model.predict(input_data)[0]

    revenue = predicted_sales * predicted_quantity

    # Revenue Growth %
    growth_percent = ((revenue - base_revenue) / base_revenue) * 100

    return render_template(
        "promotion.html",
        sales=round(predicted_sales, 2),
        quantity=round(predicted_quantity, 2),
        revenue=round(revenue, 2),
        growth=round(growth_percent, 2)
    )




# =========================================================
# FORECAST
# =========================================================
@app.route("/forecast")
def forecast():
    return render_template("forecast.html")


@app.route("/generate_forecast", methods=["POST"])
def generate_forecast():

    days = int(request.form["days"])
    forecast_values = forecast_model.forecast(steps=days)

    lower = forecast_values * 0.9
    upper = forecast_values * 1.1

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, days+1), forecast_values, marker='o')
    plt.fill_between(range(1, days+1), lower, upper, alpha=0.2)

    plt.title("Demand Forecast (Next Days)")
    plt.xlabel("Days")
    plt.ylabel("Sales")
    plt.grid(True)

    img = BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    forecast_list = [
        {"day": i+1, "value": round(val, 2)}
        for i, val in enumerate(forecast_values)
    ]

    return render_template(
        "forecast.html",
        forecast=forecast_list,
        graph=graph_url
    )


# =========================================================
# NLP
# =========================================================
@app.route("/nlp")
def nlp():
    return render_template("nlp.html")


@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment():

    review = request.form["review"]

    review_vec = vectorizer.transform([review])

    prediction = nlp_model.predict(review_vec)[0]
    probability = nlp_model.predict_proba(review_vec)[0][prediction]

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

    return render_template(
        "nlp.html",
        sentiment=sentiment,
        probability=round(probability * 100, 2)
    )


# =========================================================
# ABOUT
# =========================================================
@app.route("/about")
def about():
    return render_template("about.html")


# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
