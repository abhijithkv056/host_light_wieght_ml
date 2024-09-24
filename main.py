import uvicorn
from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel
import joblib

app = FastAPI()
predictor = joblib.load('model_temp.joblib')

class Model(BaseModel):
    one: float
    two: float
    three: float

data = Model(one=1.0, two=2.0, three=3.0)
data_dict = data.model_dump()   

item =[3,2,5,7,8]

@app.get("/")
def read_root():
    return {"message": data_dict}

class Model(BaseModel):
    one: float
    two: float
    three:float



# Define the route for prediction
@app.post('/predict')
def predict_stuff(data: Model):
    # Convert the Pydantic model to a dictionary
    data_dict = data.model_dump()
    # Debugging: print out the dictionary
    print(data_dict)
    print('one:', data_dict['one'])
    print('two:', data_dict['two'])
    print('three:', data_dict['three'])

    # Assuming you have a predictor object set up to handle predictions
    
    return {'predictor':predictor.predict([[data_dict['one'], data_dict['two'], data_dict['three']]]).tolist()}


if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
