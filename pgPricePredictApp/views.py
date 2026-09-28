import os

from sklearn.metrics import r2_score
import pandas as pd

from django.conf import settings
from django.shortcuts import render

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def predict_price(request):

    # Load dataset
    df = pd.read_csv("pg_price_prediction_dataset_1850.csv")

    # Features and target
    X = df.drop("rent", axis=1)
    y = df["rent"]

    # Categorical columns
    categorical_features = [
        "city",
        "area",
        "room_type",
        "food",
        "wifi",
        "ac",
        "attached_bathroom",
        "furnished",
        "parking",
        "cctv",
        "power_backup"
    ]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    # Preprocessor
    process = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ],
        remainder="passthrough"
    )

    # Model
    model = Pipeline([
        ("preprocessor", process),
        ("regressor", DecisionTreeRegressor(random_state=42))
    ])

    # Train model
    model.fit(X_train, y_train)

    # Test data prediction
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)

    # POST request
    if request.method == "POST":

        city = request.POST.get("city")
        area = request.POST.get("area")
        room_type = request.POST.get("room_type")
        sharing = int(request.POST.get("sharing"))
        room_size_sqft = float(request.POST.get("room_size_sqft"))
        food = request.POST.get("food")
        wifi = request.POST.get("wifi")
        ac = request.POST.get("ac")
        attached_bathroom = request.POST.get("attached_bathroom")
        furnished = request.POST.get("furnished")
        parking = request.POST.get("parking")
        cctv = request.POST.get("cctv")
        power_backup = request.POST.get("power_backup")

        # Input data
        input_data = pd.DataFrame({
            "city": [city],
            "area": [area],
            "room_type": [room_type],
            "sharing": [sharing],
            "room_size_sqft": [room_size_sqft],
            "food": [food],
            "wifi": [wifi],
            "ac": [ac],
            "attached_bathroom": [attached_bathroom],
            "furnished": [furnished],
            "parking": [parking],
            "cctv": [cctv],
            "power_backup": [power_backup]
        })
        
        
        # Input Summery
        input_summary = {
            "city":city,
            "Area": area,
            "Room Type": room_type,
            "Sharing": sharing,
            "Room Size": f"{room_size_sqft} SQFT",
            "Food": food,
            "WiFi": wifi,
            "AC": ac,
            "Attached Bathroom": attached_bathroom,
            "Furnished": furnished,
            "Parking": parking,
            "CCTV": cctv,
            "Power Backup": power_backup,
        }

        # Prediction
        result = model.predict(input_data)

        # R2 percentage
        r2_percentage = f"{r2 * 100:.2f}%"

        return render(
            request,
            "prediction.html",
            {
                "result": result,
                "r2_percentage": r2_percentage,
                "input_summary":input_summary,
            }
        )

    return render(request, "prediction.html")

