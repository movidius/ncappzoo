import tensorflow as tf
import numpy as np
import cv2


def central_crop(image: np.ndarray, central_fraction):
    """
    (Same as tf.image.central_crop but implemented in numpy)

    Crop the central region of the image.

    Remove the outer parts of an image but retain the central region of the image
    along each dimension. If we specify central_fraction = 0.5, this function
    returns the region marked with "X" in the below diagram.

         --------
        |        |
        |  XXXX  |
        |  XXXX  |
        |        |   where "X" is the central 50% of the image.
         --------

    Args:
      image: 3-D float array of shape [height, width, depth]
      central_fraction: float (0, 1], fraction of size to crop

    Raises:
      ValueError: if central_crop_fraction is not within (0, 1].

    Returns:
      3-D float array
    """

    if central_fraction <= 0.0 or central_fraction > 1.0:
        raise ValueError('central_fraction must be within (0, 1]')
    if central_fraction == 1.0:
        return image

    img_shape = image.shape
    depth = img_shape[2]
    img_h = float(img_shape[0])
    img_w = float(img_shape[1])
    y0 = int((img_h - img_h * central_fraction) / 2)
    x0 = int((img_w - img_w * central_fraction) / 2)

    box_h = img_shape[0] - y0 * 2
    box_w = img_shape[1] - x0 * 2

    image = image[y0: y0 + box_h, x0:x0 + box_w].copy()
    return image


def preprocess_for_eval_py(image, height, width, central_fraction=0.875, scope=None):
    """ Same as preprocess_for_eval but not using TF ops"""
    if central_fraction:
        image = central_crop(image, central_fraction=central_fraction)

    if height and width:
        # Resize the image to the specified height and width.
        image = cv2.resize(image, (height, width), interpolation=cv2.INTER_CUBIC)
    image = image.astype('f4')
    image /= 255.
    image -= 0.5
    image *= 2.0
    return image


def preprocess_train_cifar10(image, meta, h, w):
    # random crop
    image = np.pad(image, [(4, 4), (4, 4), (0, 0)], mode='constant', constant_values=0)
    x0 = np.random.randint(0, 8)
    y0 = np.random.randint(0, 8)
    image = image[y0:(y0+h), x0:(x0+w), :]

    # random fliprl
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)

    image = image.astype('f4')
    image /= 255.

    # norm
    image -= 0.5
    image *= 2
    return image


def preprocess_test_cifar10(image, meta, h, w):
    image = image.astype('f4')
    image /= 255.

    # norm
    image -= 0.5
    image *= 2
    return image


def preprocess_train(image, meta, h, w):
    # resize to size/.875
    image = cv2.resize(image, (int(w / .875), int(h / .875)), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float32)

    # order of operations
    order = np.arange(0, 3)
    np.random.shuffle(order)

    for i in order:
        if i == 0:
            # hue (+- 20%)
            # x = 1 + (np.random.random() * 0.4 - 0.2)
            # image[:, :, 0] *= x
            pass

        if i == 1:
            # brightness (+- 32)
            x = int(round(np.random.random() * 64 - 32))
            image[:, :, 1] += x
        if i == 2:
            # saturation (+- 50%)
            x = 1 + (np.random.random() - 0.5)
            image[:, :, 2] *= x

    # TODO: not applying tf.image.random_contrast

    # clip
    image[:, :, 0] = np.clip(image[:, :, 0], 0, 180)
    image[:, :, 1] = np.clip(image[:, :, 1], 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2], 0, 255)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HLS2RGB)

    # random fliprl
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)

    # random crops
    dh, dw = image.shape[:2]
    dh -= h
    dw -= w
    x = np.random.randint(0, dw)
    y = np.random.randint(0, dh)
    image = image[y: y + h, x:x + w].astype('f4')

    # norm to [-1, 1]
    image /= 255
    image -= 0.5
    image *= 2
    return image


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def one_hot(labels, classes, dtype='f4'):
    n = len(labels)
    out = np.zeros((n, classes), dtype)
    out[np.arange(0, n), labels] = 1
    return out


def preprocess_eval(image, meta, h, w):
    return preprocess_for_eval_py(image, h, w)
