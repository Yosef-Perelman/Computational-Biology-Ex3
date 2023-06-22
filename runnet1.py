import json
import numpy as np

with open('wnet1.json', 'r') as f:
    data = json.load(f)

network_structure = data["network_structure"]
first_layer_size = network_structure["first_layer_size"]
middle_layer_size = network_structure["middle_layer_size"]
last_layer_size = network_structure["last_layer_size"]
weights = data["weights"]
weights1 = weights["weights1"]
weights2 = weights["weights2"]

with open('output1.txt', 'w') as f:
    with open('testnet1.txt', 'r') as file:
        for line in file:
            sample = line.strip()
            sample = np.array([int(bit) for bit in sample])
            input_layer = np.concatenate((sample, np.ones(1)))

            middle_layer = np.dot(input_layer, weights1)
            middle_layer_after_activation = np.maximum(middle_layer, 0)

            last_layer = np.dot(middle_layer_after_activation, weights2)
            last_layer_after_activation = (1 / (1 + np.exp(-last_layer)))

            prediction = int(last_layer_after_activation[0] >= 0.5)
            f.write(f"{prediction}\n")