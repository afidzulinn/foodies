from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
import io
import uvicorn
from PIL import Image

from utils.processing import prepare_image

app = FastAPI()

# Load the model
model = load_model("model_acc_97.h5")

# define class name
class_names = ["bakso", "gado", "gudeg", "rendang", "sate"]


@app.get("/")
async def home():
    return {"message": "API Predict Makanan Bakso, Rendang, Gado - gado, Sate, Gudeg"}

@app.post("/predict-image")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img_array = prepare_image(img)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]

    return JSONResponse({"prediction": predicted_class})

if __name__ == "__main__":
    uvicorn.run(app, port=9786) # ini port
