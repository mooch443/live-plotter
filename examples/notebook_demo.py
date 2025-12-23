import numpy as np
from live_plotter import loop

def frames():
    for i in range(200):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:, : i % 320, 1] = 255
        yield img

loop(frames(), quality=70, max_side=1024)
