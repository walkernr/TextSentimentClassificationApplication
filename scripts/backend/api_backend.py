from pathlib import Path
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from text_classifier_predictor import text_classifier_predictor

text_path = Path(__file__).parent / '../../model/vocab/text_vocab.pt'
label_path = Path(__file__).parent / '../../model/vocab/label_vocab.pt'
model_path = Path(__file__).parent / '../../model/model.pt'
weight_path = Path(__file__).parent / '../../model/weight/weight.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
predictor = text_classifier_predictor(device, text_path, label_path, model_path, weight_path)

app = FastAPI()
origins = ["http://localhost:3000", "localhost:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def ping():
    return {"message": "welcome. append url with '/predict/{text}' for predictions"}


@app.get("/predict/{text}")
def predict(text: str):
    prediction = predictor.predict(text)
    return {text: prediction}

negative_text = 'it was bad'
positive_text = 'it was good'
predictions = [{"id": 0, "item": '{}: {}'.format(negative_text, predictor.predict(negative_text))},
               {"id": 1, "item": '{}: {}'.format(positive_text, predictor.predict(positive_text))}]

@app.get("/prediction", tags=["predictions"])
async def get_predictions() -> dict:
    return { "data": predictions }


@app.post("/prediction", tags=["predictions"])
async def add_prediction(prediction: dict) -> dict:
    prediction['item'] = '{}: {}'.format(prediction['item'], predictor.predict(prediction['item']))
    predictions.append(prediction)
    return {"data": {"prediction added"}}