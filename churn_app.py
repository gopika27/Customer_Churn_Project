from fastapi import FastAPI, Request
import pandas as pd
import pickle
import os

app = FastAPI()

# Load model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)


@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        print("Incoming data:", data)

        input_df = pd.DataFrame([data])
        print("Before dummies:", input_df)

        input_df = pd.get_dummies(input_df)
        print("After dummies:", input_df)

        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
        print("Final input:", input_df)

        prediction = model.predict(input_df)

        return {"prediction": int(prediction[0])}

    except Exception as e:
        print("ERROR:", str(e))   # 👈 VERY IMPORTANT
        return {"error": str(e)}