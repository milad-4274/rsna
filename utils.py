import pydicom
import matplotlib.pyplot as plt
from pathlib import Path

def load_jpeg2_image(path : Path):
    if not path.is_file():
        raise FileNotFoundError(f"there is no image in {path}")

    image = pydicom.dcmread(path).pixel_array
    return image


def plt_imshow(image):
    plt.figure(figsize=(10,7))
    plt.imshow(image)
    plt.axis(False)
    plt.show()

