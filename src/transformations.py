import math
import cv2
import torch
import numpy as np
import torchvision.transforms as T

toPIL = T.ToPILImage()

def get_gradient_2d(start, end, width, height, is_horizontal):
    """
    Generate a 2D gradient array
    :param start: The starting value of the gradient
    :param end: The ending value of the gradient
    :param width: The width of the resulting gradient array
    :param height: The height of the resulting gradient array
    :param is_horizontal: Determines if the gradient should be horizontal (True) or vertical (False)
    :return: 2D gradient array of shape (height, width), with values ranging from start to end
    """

    if is_horizontal:
        return np.tile(np.linspace(start, end, width), (height, 1))
    else:
        return np.tile(np.linspace(start, end, height), (width, 1)).T

def get_gradient_3d(width, height, start_list, end_list, is_horizontal_list):
    """
    Generate a 3D gradient array
    :param width: The width of the resulting gradient array
    :param height: The height of the resulting gradient array
    :param start_list: The list of starting values for each channel of the gradient
    :param end_list: The list of ending values for each channel of the gradient
    :param is_horizontal_list: The list of booleans indicating whether each channel should have a horizontal (True) or vertical (False) gradient
    :return: 3D gradient array of shape (height, width, num_channels), with values ranging from start_list to end_list
    """
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, end, is_horizontal) in enumerate(zip(start_list, end_list, is_horizontal_list)):
        result[:, :, i] = get_gradient_2d(start, end, width, height, is_horizontal)

    return result

def create_cross_mask_3d(height, width, d):
    """
    Creates a cross-shaped mask with gradients
    :param height: The height of the mask
    :param width: The width of the mask
    :param d: The thickness of the cross-shaped gradient
    :return: Cross-shaped mask of shape (height, width, channeln)
    """
    start = (0, 0, 0)
    end = (255, 255, 255)
    h_c, w_c = int(height / 2), int(width / 2)
    mask_h = np.full((height, width, 3), 255)
    mask_h[:height, w_c - d: w_c] = get_gradient_3d(d, height, end, start, (True, True, True))
    mask_h[:height, w_c: w_c + d] = get_gradient_3d(d, height, start, end, (True, True, True))
    mask_v = np.full((height, width, 3), 255)
    mask_v[h_c - d: h_c, :width] = get_gradient_3d(width, d, end, start, (False, False, False))
    mask_v[h_c: h_c + d, :width] = get_gradient_3d(width, d, start, end, (False, False, False))
    return np.minimum(mask_h, mask_v)

def create_bounding_mask_3d(height, width, d):
    """
    Creates a 3D bounding mask
    :param height: The height of the mask
    :param width: The width of the mask
    :param d: The thickness of the bounding mask
    :return: 3D bounding mask of shape (height, width, 3)
    """
    start = (0, 0, 0)
    end = (255, 255, 255)
    h_c, w_c = int(height / 2), int(width / 2)
    mask_h = np.full((height, width, 3), 255)
    mask_h[:height, :d] = get_gradient_3d(d, height, start, end, (True, True, True))
    mask_h[:height, width - d:width] = get_gradient_3d(d, height, end, start, (True, True, True))
    mask_v = np.full((height, width, 3), 255)
    mask_v[:d, :width] = get_gradient_3d(width, d, start, end, (False, False, False))
    mask_v[height - d:height, :width] = get_gradient_3d(width, d, end, start, (False, False, False))
    return np.minimum(mask_h, mask_v)

def blend_with_background(image, bg_image):
    """
    Blends image's central cross lines with background
    :param image: The input image
    :param bg_image: The background image
    :return: Blended image
    """

    blured_bg_image = cv2.GaussianBlur(bg_image,(21,21),cv2.BORDER_DEFAULT)
    mask = create_cross_mask_3d(image.shape[0], image.shape[1], 32)
    blended_image = mask / 255 * image + (1 - mask / 255) * blured_bg_image
    return blended_image.astype(np.uint8)

def randomPerspective(image, seed):
    """
    Applies a random perspective transformation to the input image
    :param image: The input image
    :param seed: The random seed used for generating the perspective transformation
    :return: The transformed image after applying the random perspective transformation
    """
    torch.manual_seed(seed)

    border = (80, 80, 80, 80)
    image_with_border = cv2.copyMakeBorder(image, *border, cv2.BORDER_REFLECT, None).astype(np.uint8)
    h, w, _ = image_with_border.shape

    perspective_transformer = T.RandomPerspective(distortion_scale=0.3, p=1.0)
    image_with_border = np.array(perspective_transformer(toPIL(image_with_border)))
    transformed_image = image_with_border[border[0]:h - border[0], border[0]:w - border[0]]
    return transformed_image


def randomAffine(image, bg_image, seed):
    """
    Applies a random affie transformation to the input image
    :param image: The input image
    :param bg_image: The background image
    :param seed: The random seed used for generating the affie transformation
    :return: The transformed image after applying the random affie transformation
    """
    torch.manual_seed(seed)

    border = (80, 80, 80, 80)
    image_with_border = cv2.copyMakeBorder(image, *border, cv2.BORDER_REFLECT).astype(np.uint8)
    h, w, _ = image_with_border.shape
    bounding_mask = create_bounding_mask_3d(h, w, 32).astype(np.uint8)

    affine_transfomer = T.RandomAffine(degrees=(20, 70), translate=(0.0, 0.0), scale=(1.0, 1.0), shear=(0, 30))
    params = affine_transfomer.get_params(affine_transfomer.degrees, affine_transfomer.translate, affine_transfomer.scale, affine_transfomer.shear, (h, w))

    image_with_border = np.array(T.functional.affine(toPIL(image_with_border), angle=params[0], translate=params[1], scale=params[2], shear=params[3]))
    bounding_mask = np.array(T.functional.affine(toPIL(bounding_mask), angle=params[0], translate=params[1], scale=params[2], shear=params[3]))

    blured_bg_image = cv2.GaussianBlur(bg_image,(21,21),cv2.BORDER_DEFAULT)
    resizedbg = cv2.resize(blured_bg_image, (h, w))

    transformed_image = bounding_mask / 255 * image_with_border + (1 - bounding_mask / 255) * resizedbg
    transformed_image = transformed_image[border[0]:h - border[0], border[0]:w - border[0]].astype(np.uint8)
    return transformed_image
