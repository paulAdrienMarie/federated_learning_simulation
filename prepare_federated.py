import json
import os

def prepare_for_federated_learning():

    with open("./static/train.json") as f:
        train_data = json.loads(f.read())
        
    with open("./static/train_base64images.json") as f:
        train_base64_data = json.loads(f.read())
        
    assert len(train_data) == len(train_base64_data), f"Inconsistent sizes: train_data has {len(train_data)} items, train_base64_data has {len(train_base64_data)} items."

    NUMUSERS = 200
    BATCHSIZE = 14

    output_dir = "./static/dataset/"
    os.makedirs(output_dir, exist_ok=True)

    for user_id in range(NUMUSERS):
        start_index = user_id * BATCHSIZE
        end_index = start_index + BATCHSIZE
        
        if start_index >= len(train_data):
            break
        
        print(f"Creating JSON file for user {user_id + 1}")
        
        pictures_ids = list(train_data.keys())[start_index:end_index]
        pictures_labels = list(train_data.values())[start_index:end_index]
        pictures_base64 = [train_base64_data.get(id) for id in pictures_ids]
        
        ds = []
        for i, id in enumerate(pictures_ids):
            if pictures_base64[i] is None:
                print(f"Warning: Missing base64 data for image ID {id}")
            ds.append({
                "id": id,
                "label": pictures_labels[i],
                "base64": pictures_base64[i]
            })
        
        output_file = os.path.join(output_dir, f"user_{user_id + 1}.json")
        print(f"Saving the set of images in {output_file}")
        
        with open(output_file, "w") as f:
            json.dump(ds, f)
            
if __name__ == "__main__":
    prepare_for_federated_learning()
