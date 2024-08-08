import json
import onnx
import onnxruntime as ort
import numpy as np
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from datasets import load_dataset

class Evaluate:
    """
    Evaluates the performances of the updated model resulting
    from the Federated Learning scenario
    
    Attributes:
    BASE_MODEL_PATH -- The relative path to the onnx base model
    UPDATED_MODEL_PATH -- The relative path to the onnx updated model
    TRUE_LABELS_PATH -- The relative path to the json file containing the true labels
    DATASET -- The dataset used, here COCO
    """
    
    def __init__(self, args):
        self.args = args
        self.BASE_MODEL_PATH = "./model/base_model.onnx"
        self.UPDATED_MODEL_PATH = "./model/inference.onnx"
        self.TRUE_LABELS_PATH = "./test.json"
        self.DATASET = "detection-datasets/coco"

    def load_model(self, path):
        """
        Loads an onnx model at a given path
        
        Arguments:
        path -- Relative path to the onnx model file
        """
        return onnx.load(path)
    
    def load_images(self):
        """
        Loads the images to compute testing
        """
        ds = load_dataset(self.DATASET)
        
        test = self.loadJson(self.TRUE_LABELS_PATH)
        
        ids = list(test.keys())
        images = [img["image"] for img in ds["val"] if str(img["image_id"]) in ids]
       
        assert len(images) == len(ids), f"Got inconsistent lenght, images got length : {len(images)} and ids got length : {len(ids)}" 
       
        return {id_: img for id_, img in zip(ids, images)}
        
    def check_model(self, model):
        """
        Checks if the downloaded model is correct
        
        Argument:
        model -- The onnx model to check
        """
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as e:
            print(f"The model is invalid: {e}")
        else:
            print("The exported model is valid!")
            
    def loadJson(self, path):
        """
        Loads a json file at a given path
        
        Arguments:
        path -- Relative path to the json file
        """
        with open(path) as f:
            return json.loads(f.read())
        
    def save_to_json(self, dictionary, path):
        """
        Saves a dict in a json file at a specified path
        
        Arguments:
        dict -- The dictionnary to be saved
        path -- The path of the json file
        """
        with open(path,"w") as f:
            json.dump(dictionary,f)
        
    def preprocess_image(self, img):
        """
        Prepares image for inference according to the onnx model input format.
        Resizes to shape [1, 3, 224, 224] and normalizes with the following parameters:
            - mean : 0.5
            - std : 0.5

        Arguments:
        img -- The image to prepare for inference (PIL Image)
        """
        PREPROCESS_CONFIG = "./static/preprocessor_config.json"
        pre = self.loadJson(PREPROCESS_CONFIG)
        input_size = (pre["size"]["height"], pre["size"]["width"])

        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.Lambda(lambda img: img.convert("RGB")),  # Convert image to RGB if it isn't already
            transforms.ToTensor(),
            transforms.Normalize(mean=pre["image_mean"], std=pre["image_std"]),
        ])

        image = transform(img).unsqueeze(0)  # Add batch dimension
        return image.numpy()

    def run_inference(self, path, image):
        """
        Runs inference using Inference Session of onnxruntime API
        
        Arguments:
        path -- The path to the onnx model
        image -- The preprocessed image ready for inference
        """
        session = ort.InferenceSession(path)  
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        results = session.run([output_name], {input_name: image})
        return results[0]
    
    def softmax(self, array):
        """
        Performs softmax activation on an array Object
        
        Arguments:
        array -- Raw outputs of the model
        """
        probabilities = np.exp(array) / np.sum(np.exp(array), axis=1, keepdims=True)
        return probabilities
    
    def predict(self, img, path):
        """
        Returns the predicted label of the given image using the given model
        
        Arguments:
        img -- The image on which to predict a label (PIL Image)
        path -- The path to the onnx model to use
        """
        CONFIG_PATH = "./static/config.json"
        id2label = self.loadJson(CONFIG_PATH)["id2label"]
        # Preprocess the image directly
        preprocessed_image = self.preprocess_image(img)
        output = self.run_inference(path, preprocessed_image)
        probabilities = self.softmax(output)[0]
        sorted_indices = np.argsort(-probabilities)[0]
        return id2label[str(sorted_indices)]
    
    def compute_accuracy(self, preds):
        """
        Computes accuracy score
        
        Arguments:
        preds -- The predictions of a model as a list
        """
        Y_true = self.loadJson(self.TRUE_LABELS_PATH)
        return accuracy_score(list(Y_true.values()), preds)
        
    def __call__(self):
        """
        Runs inference on both base model and updated model
        and computes accuracy score
        """
        self.check_model(self.load_model(self.BASE_MODEL_PATH))
        self.check_model(self.load_model(self.UPDATED_MODEL_PATH))

        images = self.load_images()
        
        base_model_output = {}
        updated_model_output = {} 
        count = 1
        
        for id, img in images.items():
            print(f"Inference running on image {count}/{len(images)}")
            # run inference on the base model
            label = self.predict(img, self.BASE_MODEL_PATH)
            base_model_output[int(id)] = label
            # run inference on the updated model
            label = self.predict(img, self.UPDATED_MODEL_PATH)
            updated_model_output[int(id)] = label
            count += 1
            
        self.save_to_json(base_model_output,"base_predictions.json")
        self.save_to_json(updated_model_output,"updated_predictions.json")
            
        print(f"Accuracy of the base model on test set: {self.compute_accuracy(list(base_model_output.values()))}")
        print(f"Accuracy of the updated model on test set: {self.compute_accuracy(list(updated_model_output.values()))}")

        
if __name__ == "__main__":
    evaluate = Evaluate(args="Evaluate base model and updated model")
    evaluate()
