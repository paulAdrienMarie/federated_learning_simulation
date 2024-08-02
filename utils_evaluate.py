from PIL import Image
import os
import numpy as np
import json
import torchvision.transforms as transforms
import onnxruntime as ort
import onnx
from io import BytesIO
import base64

with open("./static/preprocessor_config.json") as f:
    pre = json.loads(f.read())
    
with open("./static/config.json") as f:
    id2label = json.loads(f.read())["id2label"]
    
def check_model(model_path):
    model = onnx.load(model_path)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: {}".format(e))
    else:
        print("The exported model is valid!")

def load_from_path(dir):
        
    IMAGES_DIR = dir
    images = {}
    
    with open("./cache.json") as f:
        data = json.load(f)
    
    skip_keys = [key for key in data.keys() if data.get(key).lower() == "i don't know" or data.get(key).lower() == "none"]

    for file_name in os.listdir(IMAGES_DIR):
        
        file_path = os.path.join(IMAGES_DIR,file_name)
        
        if not (os.path.splitext(file_path)[1] == ".png") or (os.path.splitext(file_path)[0] in skip_keys):
            pass
        else:
            key = os.path.splitext(file_path)[0]
            images[key] = Image.open(file_path).convert("RGB")
            
    return images
    
def preprocess_image(img):
    input_size = (pre["size"]["height"], pre["size"]["width"])
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=pre["image_mean"], std=pre["image_std"]),
        ]
    )
    image = transform(img).unsqueeze(0)  # Add batch dimension
    return image.numpy()

def data_augmentation(img):
    
    num_images = 10
    images = [] 
    input_size = (pre["size"]["height"], pre["size"]["width"])

    for i in range(num_images):
    
        transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=pre["image_mean"], std=pre["image_std"]),
            ]
        )
        image = transform(img).unsqueeze(0)
        image = image.numpy()
        images.append(image)
    
    return images

def run_inference(base64_data, model_path):
    
    session = ort.InferenceSession(model_path)  
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    image = preprocess_image(base64_data)
    results = session.run([output_name], {input_name: image})
    

    return results[0]

def softmax(z):
    probabilities = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return probabilities

def generate_target_logits(target_class_index, num_classes, large_value, low_value):
    logits = np.full((1, num_classes), low_value, dtype=np.float32)
    logits[0][target_class_index] = large_value
    
    return logits

def get_labels(base64_data, model_path):
    
    output = run_inference(base64_data, model_path)
    probabilities = softmax(output)[0]
    
    sorted_indices = np.argsort(-probabilities)
    
    labels = {}
    for x, i in enumerate(sorted_indices):
        if x > 4:
            break
        labels[id2label[str(i)]] = str(probabilities[i])
    
    return labels