from fastapi import FastAPI , File , UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

templates = Jinja2Templates(directory="templates")
MODEL = tf.keras.models.load_model("potatoes.h5")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "hello , I am alive"


def read_file_as_image(data) -> np.ndarray:
   image = np.array(Image.open(BytesIO(data)))
   return image

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})

@app.post("/predict")
async def predict(

    file: UploadFile = File(...)
):
    image =read_file_as_image(await file.read())

    image_batch = np.expand_dims(image, 0) 
    predictions = MODEL.predict(image_batch)
    pred_class= CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class' : pred_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app , host='127.0.0.1' , port = 8000 )