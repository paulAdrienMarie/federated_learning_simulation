import json

def filter():
    # Define the file path
    cache_file_path = "./cache.json"

    # Open and load the cache JSON file
    with open(cache_file_path) as f:
        cache = json.load(f)

    # Extract values from the cache and filter them
    filtered_dict = {key: value for key, value in cache.items() if value.lower() != "i don't know" and value.lower() != "none"}
    
    # Save the filtered dictionary to a new JSON file
    with open("filtered_dataset.json", "w") as f:
        json.dump(filtered_dict, f, indent=4)

    # Calculate filtered values
    filtered_values = [value for value in cache.values() if value.lower() == "i don't know" or value.lower() == "none"]

    # Print statistics
    print("LENGTH OF CACHE = {}".format(len(cache)))
    print("NB OCCURRENCES OF 'I DON'T KNOW' OR 'NONE' = {}".format(len(filtered_values)))
    print("NB WELL PREDICTED = {}".format(len(cache) - len(filtered_values)))

if __name__ == "__main__":
    filter()
