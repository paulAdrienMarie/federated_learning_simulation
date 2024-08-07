from base64images import Base64Generator
from openai_request import AltGenerator
from utils_dataset import DatasetCleaner
import argparse

class Pipeline:
    """
    Handles the whole process of the dataset creation:
    
        - Generates the label for every image of the dataset
        - Removes unwanted patterns from the resulting json file
        - Generates the string representation of the images using base64 encoding
        
    Attributs:
    option -- The set of images to handle - can be either train or set
    AltGenerator -- Instance of the AltGenerator class to generate label
    DatasetCleaner -- Instance of the DatasetCleaner class to clean the dataset
    Base64Generator -- Instance of the Base64Generator class 
    """
    
    def __init__(self, option):
        """
        Initializes a new instance of the Pipeline class
        
        Arguments:
        option -- The set of images to handle - can be either train or test
        """
        
        self.option = option,
        self.AltGenerator = AltGenerator(args="Image classification using gpt-4o", option=option)
        self.DatasetCleaner = DatasetCleaner(option=option) 
        self.Base64Generator = Base64Generator(args="Generate the base64 string reprentation of a set of images")
        
    def __call__(self):
        """
        Calls __call__ functions of each class
        """
        
        self.AltGenerator()
        self.DatasetCleaner()
        self.Base64Generator()
        
if __name__=="__main__":
    parser = argparse.ArgumentParser(prog="Make Dataset")
    parser.add_argument("option", type=str, help="Dataset Option")
    arg = parser.parse_args()
    
    pipeline = Pipeline(arg.option)
    # Launch the pipeline
    pipeline()