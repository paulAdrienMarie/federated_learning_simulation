from aiohttp import web
import os
import onnx
import numpy as np
from onnx import numpy_helper
from artifacts import gen_artifacts


HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE,"model/inference.onnx")

model = onnx.load(MODEL_PATH)

weights = {}

async def update_model(request):
    
    try:
        data = await request.json()
        
        updated_weights = data.get("updated_weights")
        user_id = data.get("user_id")
        
        print('RECEIVED DATA FOR USER NUMBER {}'.format(user_id))
        print(user_id<2)
        
        if user_id < 100:
        
            updated_weights_list = list(updated_weights["cpuData"].values())
            updated_weights_array = np.array(updated_weights_list, dtype=np.float32).reshape((1000,768))
            weights[user_id] = updated_weights_array.astype(np.float32)
            print("DATA LENGTH {}".format(len(weights)))
        
        else:
            
            print("UPDATING THE MODEL")
            NBUSER = len(weights.keys())
    
            # Assuming all weight matrices have the same shape
            first_key = next(iter(weights))
            weight_shape = weights[first_key].shape

            # Create a 3D array to hold all user weights
            weights_array = np.empty((NBUSER, *weight_shape))

            for i, value in enumerate(weights.values()):
                weights_array[i] = value

            # Compute the average across the user dimension (axis=0)
            averaged_weights_array = np.mean(weights_array, axis=0)
            
            CLASSIFIER_WEIGHT = averaged_weights_array.astype(np.float32)
                                
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
    print("NEW CONNECTION")
    return web.FileResponse("./index.html")

async def style(request):
    return web.FileResponse("./style.css")