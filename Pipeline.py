from Base64Generator import Base64Generator
from LabelGenerator import AltGenerator
from DatasetCleaner import DatasetCleaner
import argparse

class Pipeline:
    """
    Handles the whole process of dataset creation:
    
        - Generates the label for every image of the dataset.
        - Removes unwanted patterns from the resulting json file.
        - Generates the string representation of the images using base64 encoding.
        
    Attributes:
    option -- The set of images to handle - can be either 'train' or 'test'.
    AltGenerator -- Instance of the AltGenerator class to generate labels.
    DatasetCleaner -- Instance of the DatasetCleaner class to clean the dataset.
    Base64Generator -- Instance of the Base64Generator class.
    """
    
    def __init__(self, option):
        """
        Initializes a new instance of the Pipeline class.
        
        Arguments:
        option -- The set of images to handle - can be either 'train' or 'test'.
        """
        self.option = option
        self.AltGenerator = AltGenerator(args="Image classification using gpt-4", option=self.option)
        self.DatasetCleaner = DatasetCleaner(option=option) 
        self.Base64Generator = None
        
    def __call__(self):
        """
        Calls __call__ functions of each class.
        """
        
        if self.option == "train":
            self.AltGenerator()
            self.DatasetCleaner()
            self.Base64Generator = Base64Generator(args="Generate the base64 string representation of a set of images")
            self.Base64Generator()
        else:
            self.AltGenerator()
            self.DatasetCleaner()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Make Dataset")
    parser.add_argument("option", type=str, help="Dataset Option (train/test)")
    args = parser.parse_args()
    if not args.option == "train" and not args.option == "test":
        exit(
            print("Option must be train or test")
        )
        
    pipeline = Pipeline(args.option)
    # Launch the pipeline
    pipeline()
