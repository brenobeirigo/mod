import numpy as np


def movingaverage(data, w, start=0, start_den=2):
    new_data = np.zeros(len(data))
    for i in range(len(data)):
        if i < start:
            new_data[i] = sum(data[i: i + int(w / start_den)]) / int(
                w / start_den
            )
            continue
        if i + w < len(data):
            new_data[i] = sum(data[i: i + w]) / w
        else:
            new_data[i] = sum(data[i - w: i]) / w
    return new_data

