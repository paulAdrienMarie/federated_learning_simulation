import os
import base64
import json
import threading
from io import BytesIO
from datasets import load_dataset

HERE = os.path.dirname(__file__)
IMAGES_DIR = "./dest/"
NUM_THREADS = 20

# Load the images from the /dest folder
# We want to skip the images for which chatgpt has not been able to classify
# We also want to skip the files that are not in .png format

class Base64Generator():
    """
    Generates the string representation of a set of images using base64 encoding
    Aims to be used during the Federated Learning scenario in the web
    
    Attributs:
    args -- Description of what does the class
    """
    
    def __init__(self, args):
        """
        Initializes a new Base64Generator instance 
        
        Arguments:
        args -- Description of what does the class
        
        """
        self.args = args

    def generate(self):
        """
        Creates a dictionnary {id, string} using the the COCO dataset
        """
        
        DATASET = "detection-datasets/coco"
        ds = load_dataset(DATASET)
        images = {}

        with open("./static/train.json") as f:
            data = json.loads(f.read())
            
        keys = list(data.keys())
        chunk_size = (len(keys) + NUM_THREADS - 1) // NUM_THREADS  # Ensure chunks cover all items

        # Create and start threads
        threads = []
        results = [None] * NUM_THREADS
        for i in range(NUM_THREADS):
            start_index = i * chunk_size
            end_index = min((i + 1) * chunk_size, len(keys))
            chunk_keys = keys[start_index:end_index]
            
            thread = threading.Thread(target=self.process_chunk, args=(chunk_keys, ds, data, images, results, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # Check if the results are consistent
        assert len(images) == len(data), f"Inconsistent data length: {len(images)} for images and {len(data)} for clean_data"
        return images

    def process_chunk(self, chunk_keys, ds, images, results, index):
        """
        Process a chunk of the set of images to treat
        
        Arguments:
        chunk_keys -- List of keys for the images to process in this chunk.
        ds -- The dataset containing the images.
        images -- Shared dictionary to store processed images.
        results -- List to store local results for each thread.
        index -- Index of the current thread.
        """
        
        local_images = {}
        count = 1

        for id_ in chunk_keys:
            id = int(id_)
            print(f"Thread {index + 1} - Treating image with id {id}, {count}/{len(chunk_keys)}")
            img = [item["image"] for item in ds["val"] if item["image_id"] == id]
            if img:
                local_images[id] = self.image_message(img[0])["url"]
            count += 1

        # Store results in shared dictionary
        images.update(local_images)
        results[index] = local_images

    # Convert the images in base64 encoded format
    # This will be used in the javascript 

    def image_message(self, image):
        """
        Generates the string representation of the given image
        using base64 encoding
        
        image -- The image to process
        """
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_byte = buffered.getvalue()
        img_base64 = base64.b64encode(img_byte).decode("utf-8")
        encoded_image = f"data:image/png;base64,{img_base64}"
        
        return {"url": encoded_image}

    def __call__(self):
        """
        Runs the generate function and saves the resulting dict in 
        a json file
        """
        
        images = self.generate()
        print("IMAGES LENGTH {}".format(len(images)))
        
        with open("./static/train_base64images.json", "w") as f:
            json.dump(images, f)
        
if __name__ == "__main__":
    args = "Generate the base64 string reprentation of a set of images"
    generatore = Base64Generator(args=args)
    
    # Generate the base64 representation of the images
    generatore()
