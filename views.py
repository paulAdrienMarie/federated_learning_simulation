from aiohttp import web
import os
import onnx
import numpy as np
from onnx import numpy_helper
from artifacts import gen_artifacts


HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE,"model/inference.onnx")
i = 0

model = onnx.load(MODEL_PATH)

async def update_model(request):
    
    try:
        data = await request.json()
        
        updated_weights = data.get("updated_weights")
        user_id = data.get("user_id")
        
        print('RECEIVED DATA FOR USER {}'.format(user_id))
        
        data = {}
        
        if user_id < 100:
        
            updated_weights_list = list(updated_weights["cpuData"].values())
            updated_weights_array = np.array(updated_weights_list, dtype=np.float32).reshape((1000,768))
            data[user_id] = updated_weights_array.astype(np.float32)
            print(data)
        
        else:
            
            print(len(data.keys()))
            
            weights = np.array(data.values())
            
            CLASSIFIER_WEIGHT = np.empty(shape=(1000,768))
            
            for i in range(CLASSIFIER_WEIGHT.shape[0]):
                for j in range(CLASSIFIER_WEIGHT.shape[1]):
                    CLASSIFIER_WEIGHT[i][j] = np.mean(weights[i,j,:])
        
            for initializer in model.graph.initializer:
    
                if initializer.name == "classifier.weight":
                    # pdb.set_trace()
                    print(f'Original dims: {initializer.dims}')
                    new_weights_tensor = numpy_helper.from_array(CLASSIFIER_WEIGHT, initializer.name)
                    initializer.CopyFrom(new_weights_tensor)
            
                    onnx.save_model(model, MODEL_PATH)
        
                    data = {"message": "MODEL AND TRAINING ARTIFACTS UPDATED"}
   
                    gen_artifacts()
                    
        
        return web.json_response(data)
    
    except Exception as e:
        
        error_message = str(e)
        response_data = {"error": error_message}
        return web.json_response(response_data, status=500)


async def index(request):
    return web.FileResponse("./index.html")

async def style(request):
    return web.FileResponse("./style.css")

def federated(request):
    print("REQUEST RECEIVED")
    if (i<100):
        return web.FileResponse("./federated.js",)
    else:
        print("NUMBER OF USERS EXCEEDED")