import numpy as np
from PIL import Image

# image -> bin
def from_image(in_file: str, out_file: str):
    image = np.array(Image.open(in_file))
    image = np.dstack((image, np.zeros((image.shape[0], image.shape[1]))))
    image = image.astype(np.uint8)
    w, h = image.shape[1], image.shape[0]
    size = np.array([w, h]).astype(np.int32)
    with open(out_file, 'wb') as f:
       f.write(size.tobytes())
       f.write(image.tobytes())

# bin -> image
def to_image(in_file: str, out_file: str):
    size = np.fromfile(in_file, dtype=np.int32, count=2)
    w, h = size[0], size[1]
    with open(in_file, 'rb') as f:
        f.seek(8)
        image = np.fromfile(f, dtype=np.uint8)
    print("h =", h, "w =", w)
    image = image.reshape((h, w, 4))
    image = image[:, :, :3]
    image = Image.fromarray(image)
    image.save(out_file)

# Раскомментировать для image -> bin
# in_file = '1.png'
# out_file = '1.bin'
# from_image(in_file, out_file)

# Раскомментировать для bin -> image
in_file = 'result.bin'
out_file = 'result.png'
to_image(in_file, out_file)
