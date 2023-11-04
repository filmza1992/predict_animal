import pickle
import json

import numpy as np

animal = {
    0 : "cheetah",
    1 : "fox",
    2 : "hyena",
    3 : "lion",
    4 : "wolf",
}

def predict_animal(m,hog):
    result = m.predict(hog)
    print(result)
    result = np.argmax(result,axis=1)
    return animal[result[0]]