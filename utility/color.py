import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

input_path = "data/SD/AsphaltRoad2.png"      
output_path = "purple_bscan.png"

img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

gray = img.astype(np.float32)
amp = np.abs(gray - 128)

vmin, vmax = np.percentile(amp, [2, 99.5])
amp = np.clip(amp, vmin, vmax)

norm = (amp - vmin) / (vmax - vmin + 1e-8)

gamma = 0.75
norm = norm ** gamma


purple_cmap = LinearSegmentedColormap.from_list(
    "purple_bscan",
    [
        "#26004d",  
        "#3500b8",  
        "#0055ff",  
        "#00c8ff",  
        "#fff200",  
        "#ff2a00",  
    ],
    N=256
)


colored = purple_cmap(norm)
colored_rgb = (colored[:, :, :3] * 255).astype(np.uint8)

new_image = cv2.cvtColor(colored_rgb, cv2.COLOR_RGB2BGR)

new_image = new_image[:, 0:18000]

cv2.imwrite(output_path, new_image)

print("saved:", output_path)
