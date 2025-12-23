# live-plotter

Live-update a generator of image frames inside a Jupyter notebook cell.

## Install

```bash
pip install git+https://github.com/your-org/live-plotter.git
```

## Usage

```python
import numpy as np
from live_plotter import loop

def frames():
    for i in range(200):
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:, : i % 320, 1] = 255
        yield img

loop(frames(), quality=70, max_side=1024)
loop(frames(), image_format="png", max_side=1024, png_compress_level=3)
```

## Notes

- Frames can be HxW (grayscale) or HxWxC (RGB/RGBA).
- Float inputs are assumed to be in [0, 1] and converted to uint8.
- The newest frame always wins; older frames are dropped to avoid backlog.
