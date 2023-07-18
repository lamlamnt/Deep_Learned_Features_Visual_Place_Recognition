import subprocess
import json

# Run the Python script multiple times with different config values
config_path = "/home/lamlam/code/visual_place_recognition/clustering/config_multiple.json"
with open(config_path, 'r') as file:
    config_data = json.load(file)
    config_data["reference_run"] = 0
    config_data["query_run"] = 9
with open(config_path, 'w') as file:
    json.dump(config_data, file, indent=4)

# Run the Python script using subprocess
#subprocess.run(['python', 'get_cluster.py'])