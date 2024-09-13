import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_image(path, to_rgb=True):
    img = Image.open(path)
    return img.convert('RGB') if to_rgb else img


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def to_8b_image(image):
    return (255.* np.clip(image, 0., 1.)).astype(np.uint8)


def to_3ch_image(image):
    if len(image.shape) == 2:
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3:
        assert image.shape[2] == 1
        return np.concatenate([image, image, image], axis=-1)
    else:
        print(f"to_3ch_image: Unsupported Shapes: {len(image.shape)}")
        return image


def to_8b3ch_image(image):
    return to_3ch_image(to_8b_image(image))

idxs = [200,564,1228,1240,1252]
for idx in idxs:
    msk_path = 'datasets/AIST_mocap/d17/0/masks/frame_{:06d}.png'.format(idx)
    img_path = 'datasets/AIST_mocap/d17/0/images/frame_{:06d}.png'.format(idx)

    img = np.array(load_image(img_path))
    msk = np.array(load_image(msk_path))
    print(msk.max(), msk.min(), img.max(), img.min())

    img[msk==0] = 255.0
    save_image(img, 'vis_frame_{:06d}.png'.format(idx))