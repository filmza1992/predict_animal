from fastapi import FastAPI,Request, HTTPException
import pickle
import os
from app.code import predict_animal
from keras import layers, models 
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import load_model


app = FastAPI()

model = load_model(os.getcwd()+r'/model/AnimalImageFeatureModel.h5')
#model = load_model(r'../model/AnimalImageFeatureModel.h5')

@app.get("/")
def root():
    return {"message" : "This is my Second Container api"}

@app.post("/api/predict")
async def upload_image(request : Request): 
    try:
        item =  await request.json()
        hog = item['Hog']
        print("Start predict")
        result = predict_animal(model,[hog])
        print(result)
        return {"Result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))