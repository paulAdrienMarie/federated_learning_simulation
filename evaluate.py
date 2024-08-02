import os
from PIL import Image
from sklearn.metrics import accuracy_score
from utils_evaluate import load_from_path, get_labels, check_model
import time

# Define model paths
BASE_MODEL_PATH = "../../EVAL/model.onnx"
UPDATED_MODEL_PATH = "./model/inference.onnx"

# Check models (uncomment these if you need to check the models)
# check_model(BASE_MODEL_PATH)
# check_model(UPDATED_MODEL_PATH)

# Define image path
IMAGES_PATH = "./dataset/"
images = {}

# Function to replace underscores with spaces
def replace_underscores_with_spaces(input_string):
    return input_string.replace('_', ' ')

# Load images
for class_name in os.listdir(IMAGES_PATH):
    class_path = os.path.join(IMAGES_PATH, class_name)
    if os.path.isdir(class_path):  # Ensure it's a directory
        if '_' in class_name:
            class_name = replace_underscores_with_spaces(class_name)
            print(f"Class name after replacing underscores: {class_name}")
        print(f"Initialized list for class {class_name}")
        images[class_name] = []
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            try:
                img = Image.open(image_path).convert("RGB")
                images[class_name].append(img)
            except Exception as e:
                print(f"Failed to load image {image_path}: {e}")

# Prepare true labels and run inference
Y_true = []
predictions = []
updated_predictions = []
wrong_classifications = {}

for class_name, list_image in images.items():
    print(f"Starting inference on class '{class_name}' with {len(list_image)} images")
    wrong_classifications[class_name] = 0
    for i, img in enumerate(list_image):
        try:
            print(f"Running inference on image {i + 1}/{len(list_image)} of class '{class_name}'")
            pred_base = list(get_labels(img, BASE_MODEL_PATH))[:3]
            pred_updated = list(get_labels(img, UPDATED_MODEL_PATH))[:3]
            if class_name not in pred_base or class_name not in pred_updated:
                print(f"Base model predicted: {pred_base}, Updated model predicted: {pred_updated}")
                wrong_classifications[class_name] += 1
                img.show()
                time.sleep(5)
            predictions.append(pred_base[0])
            updated_predictions.append(pred_updated[0])
            Y_true.append(class_name)
        except Exception as e:
            print(f"Failed to run inference on image {i + 1} of class '{class_name}': {e}")

# Check for consistency in lengths
assert len(Y_true) == len(predictions), f"Inconsistent lengths between Y_true (got {len(Y_true)}) and predictions (got {len(predictions)})"
assert len(Y_true) == len(updated_predictions), f"Inconsistent lengths between Y_true (got {len(Y_true)}) and updated_predictions (got {len(updated_predictions)})"

# Calculate and print accuracy
accuracy_base = accuracy_score(Y_true, predictions)
accuracy_updated = accuracy_score(Y_true, updated_predictions)

for classe in wrong_classifications.keys():
    print(f"Class {classe} has {wrong_classifications.get(classe)} wrong classifications") 

print(f"Accuracy on the base model: {accuracy_base}")
print(f"Accuracy on the updated model: {accuracy_updated}")
