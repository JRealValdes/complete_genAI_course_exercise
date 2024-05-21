from fastapi import FastAPI
from pydantic import BaseModel
import iris_tools
import uvicorn

'''
NOTA: Levantar servidor con: uvicorn main:app --reload
'''

class PredictionRequest(BaseModel):
    input: list[float]

app = FastAPI()

@app.post("/predict")
async def predict_class(pred_request: PredictionRequest):
    prediction = iris_tools.inference(pred_request.input)
    return {"prediction": prediction}

if __name__ == "__main__":
    # test_data = [
    #     [4.6, 3.4, 1.4, 0.3], [7.2, 3.2, 6.0, 1.8], [4.8, 3.0, 1.4, 0.1], [6.1, 2.9, 4.7, 1.4], [4.9, 3.1, 1.5, 0.1],
    #     [6.4, 2.8, 5.6, 2.2], [5.0, 3.2, 1.2, 0.2], [5.8, 2.7, 4.1, 1.0], [5.1, 3.3, 1.7, 0.5], [7.1, 3.0, 5.9, 2.1],
    #     [4.8, 3.4, 1.9, 0.2], [6.2, 2.9, 4.3, 1.3], [4.9, 3.0, 1.4, 0.2], [5.6, 2.5, 3.9, 1.1], [5.0, 3.5, 1.6, 0.6],
    #     [6.3, 3.3, 4.7, 1.6], [5.1, 3.8, 1.9, 0.4], [6.9, 3.1, 5.4, 2.1], [4.7, 3.2, 1.6, 0.2], [6.7, 3.3, 5.7, 2.5],
    #     [5.9, 3.0, 4.2, 1.5], [6.8, 3.0, 5.5, 2.1], [5.8, 2.7, 3.9, 1.2], [7.3, 2.9, 6.3, 1.8], [6.0, 3.0, 4.5, 1.5],
    #     [6.7, 3.1, 5.6, 2.4], [5.9, 3.2, 4.8, 1.8], [6.5, 3.0, 5.2, 2.0], [6.1, 2.8, 4.0, 1.3], [6.3, 2.8, 5.1, 1.5]
    # ]
    my_pred_request = PredictionRequest(input=[1.0, 2.0, 3.0, 4.0])
    print(iris_tools.inference(my_pred_request.input))
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
