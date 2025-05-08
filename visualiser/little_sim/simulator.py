import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

json_path = os.path.join(os.path.dirname(__file__), '..', 'json_file', 'coordinates.json')

def circle(x: float, radius: float, center: tuple) -> tuple:
    y = center[0] + np.sqrt(radius ** 2 - (x - center[0]) ** 2)
    return y,

x = np.linspace(-5, 5, 10)
radius = 5.
center = (0, 0)

i = 0
forward = True
counter = 0
while True:
    if counter == len(x)-1:
        if forward:
            forward = False
        else:
            forward = True
        counter = 0

    y1 = circle(float(x[i]), radius, center)[0]
    y2 = y1 * -1

    if forward:
        y = y1
    else:
        y = y2


    # open json file
    with open(json_path, 'r') as file:
        try:
            data = json.load(file)
        except json.decoder.JSONDecodeError:
            data = []

    # add new coordinates
    data.append({"coordinate": (x[i], y)}, )

    # save json file
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Updated coordinates.json with coord_{i}: {x[i]}, {y}")

    if forward:
        i += 1
    else:
        i -= 1
    counter += 1
    time.sleep(1)