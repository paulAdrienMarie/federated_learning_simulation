import os
import base64
from PIL import Image
from io import BytesIO
import json

HERE = os.path.dirname(__file__)
IMAGES_DIR = "./dest/"

# Load the images from the /dest folder
# We want to skip the images for which chatgpt has not been able to classify
# We also want to skip the files that are not in .png format

def load_from_path():
        
    images = {}
    
    with open("./cache.json") as f:
        data = json.load(f)

    skip_keys = [key for key in data.keys() if data.get(key) == "I don't know"]
    
    for file_name in os.listdir(IMAGES_DIR):
        
        file_path = os.path.join(IMAGES_DIR,file_name)
        
        if not (os.path.splitext(file_path)[1] == ".png") or (os.path.splitext(file_path)[0] in skip_keys) :
            pass
        else:
            key = os.path.splitext(file_path)[0]
            images[key] = image_message(Image.open(file_path).convert("RGB"))["url"]
            
    return images

# Convert the images in base64 encoded format
# This will be used in the javascript 

def image_message(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte).decode("utf-8")
    encoded_image = f"data:image/png;base64,{img_base64}"
    
    return {"url": encoded_image}

def main():
    
    images = load_from_path()
    print("IMAGES LENGTH {}".format(len(images)))
    
    with open("./static/base64images.json","w") as f:
        json.dump(images,f)
        
         
if __name__ == "__main__":
    main()