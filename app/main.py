from pathlib import Path

from fastapi import FastAPI
import joblib
import pandas as pd

from app.schema import InputData

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent

# Load saved model + columns
model = joblib.load(BASE_DIR / "model" / "airline_model.pkl")
model_columns = joblib.load(BASE_DIR / "model" / "model_columns.pkl")
service_cols = joblib.load(BASE_DIR / "model" / "service_cols.pkl")


@app.post("/predict")
def predict(data: InputData):

    df = pd.DataFrame([data.dict()])

    # ---- encoding (same as training) ----
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})
    df['Customer_Type'] = df['Customer_Type'].map({'First-time': 0, 'Returning': 1})
    df['Type_of_Travel'] = df['Type_of_Travel'].map({'Personal': 0, 'Business': 1})

    # Class encoding (same logic as training one-hot)
    class_map = {
        "Eco": "Class_Eco",
        "Eco Plus": "Class_Eco Plus",
        "Business": "Class_Business"
    }

    for col in ["Class_Eco", "Class_Eco Plus", "Class_Business"]:
        df[col] = 0

    if data.Class in class_map:
        df[class_map[data.Class]] = 1

    # ---- age group ----
    if data.Age <= 25:
        df["Age_Group_Young"] = 1
        df["Age_Group_Adult"] = 0
        df["Age_Group_Middle_Aged"] = 0
        df["Age_Group_Senior"] = 0

    elif data.Age <= 40:
        df["Age_Group_Young"] = 0
        df["Age_Group_Adult"] = 1
        df["Age_Group_Middle_Aged"] = 0
        df["Age_Group_Senior"] = 0

    elif data.Age <= 60:
        df["Age_Group_Young"] = 0
        df["Age_Group_Adult"] = 0
        df["Age_Group_Middle_Aged"] = 1
        df["Age_Group_Senior"] = 0

    else:
        df["Age_Group_Young"] = 0
        df["Age_Group_Adult"] = 0
        df["Age_Group_Middle_Aged"] = 0
        df["Age_Group_Senior"] = 1

    # ---- derived features ----
    df["Delay_Diff"] = data.Arrival_Delay - data.Departure_Delay

    df["Total_Service_Score"] = sum([
        data.Departure_and_Arrival_Time_Convenience,
        data.Ease_of_Online_Booking,
        data.Check_in_Service,
        data.Online_Boarding,
        data.Gate_Location,
        data.On_board_Service,
        data.Seat_Comfort,
        data.Leg_Room_Service,
        data.Cleanliness,
        data.Food_and_Drink,
        data.In_flight_Service,
        data.In_flight_Wifi_Service,
        data.In_flight_Entertainment,
        data.Baggage_Handling
    ])

    # ---- align columns ----
    df = df.reindex(columns=model_columns, fill_value=0)

    # ---- prediction ----
    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction),
        "result": "Satisfied" if prediction == 1 else "Neutral or Dissatisfied"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)