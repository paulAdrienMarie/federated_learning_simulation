# SIMULATION OF FEDERATED LEARNING

This project aims to simulate a federated learning scenario on one single computer. The model used here is the Vision Transformer of google, googlevit patch16 224.

In order to do this, the users are simulated by chatgpt which provides for each image used the true label. To create this dataset, we use the OPENAI API to loop over all the images of the dataset. At each comparison, if the model prediction is not the same as the prediction of chatgpt-4o, the model is trained on the prediction of chatgpt-4o. 

The provided version runs for `100 users`. Each user has `20 images` to treat. For the training loop, for each image that needs to be trained, data augmentation is performed to provide a total of `7 images` to train on for a given label.

The training occurs in the frontend in javascriptusing the `ONNXRUNTIME JAVASCRIPT API` that handles training. The updated weights are stored in a python backend that updates the model once all the new weights have been received.

## INSTALL REQUIREMENTS   

- Create a new virtual environnement : `python3 -m venv env` and activate it `source env/bin/activate`. Install the requirements for this project `pip install -r requirements.txt`.

## RUN THE PROJECT

- To run the whole project, run `python watcher.py`

## DETAILS OF THE FILES 

### Training

- `artifacts.py` : Script to create the training artifacts. See `README.md` in the artifacts folder
- `check_conflict.py` : Script to check if there are duplicates in a dataset of images in a folder
- `make_dataset.py` : Script to create the dataset to evaluate the model


### Web Server

- `main.py` : Script to launch the python server
- `routes.py` : Script to specify the routes of the python server
- `view.py` : Script to define which function to run to handle a request on a route
- `watcher.py` : Script to restart the server when a file is modified within the project
- `openai_request.py` : Script to create the `train.json` file - dataset used within the training loop in javascript
- `base64images.py` : Script to create the `train_base64images.json` file that is used within the training loop in javascript

### Federated Learning 

- `evaluate.py` : Script to evaluate the difference between the base model and the updated model
- `test_connection_selenium.py` : Script to run the federated learning scenario in headless mode - doesn't work yet
- `utils_evaluate.py` : Functions used in the evaluate.py file
- `utils.py` : Function used to clean the dataset created by the `openai_request.py`


 
