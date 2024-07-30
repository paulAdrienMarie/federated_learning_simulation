import json

def filter():
    # Define the file path
    cache_file_path = "./cache.json"

    # Open and load the cache JSON file
    with open(cache_file_path) as f:
        cache = json.load(f)

    # Extract values from the cache and filter them
    filtered_dict = {key: value for key, value in cache.items() if value != "I don't know"}
    
    # For debugging purposes, if needed
    # pdb.set_trace() 

    with open("filtered_dataset.json", "w") as f:
        json.dump(filtered_dict, f)

    filtered_values = [value for value in cache.values() if value == "I don't know"]

    print("LENGTH OF CACHE = {}".format(len(cache)))
    print("NB OCCURENCE OF I DON'T KNOW = {}".format(len(filtered_values)))
    print("NB WELL PREDICTED = {}".format(len(cache) - len(filtered_values)))

if __name__ == "__main__":
    filter()
