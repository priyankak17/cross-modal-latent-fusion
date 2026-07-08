import argparse
from torchvision import transforms
from torchvision.utils import save_image
import os
import cv2
import numpy as np

def apply_low_light_effect(img, factor=0.3):
    """
    Apply a low light effect to an image.
    
    :param img: Input image.
    :param factor: The factor by which to reduce the brightness (default 0.3).
    :return: Low-light image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = hsv[..., 2] * factor
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def degrade_resolution(img, scale=0.5):
    """
    Degrade the resolution of an image.
    
    :param img: Input image.
    :param scale: The factor by which to reduce the resolution (default 0.5).
    :return: Low-resolution image.
    """
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    low_res = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(low_res, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

def process_image(image_path):
    """
    Process the image to simulate low-resolution and low-light conditions.
    
    :param image_path: Path to the input image.
    :return: Processed image tensor.
    """
    img = cv2.imread(image_path)
    low_light_img = apply_low_light_effect(img, factor=0.3)
    low_res_img = degrade_resolution(low_light_img, scale=0.5)
    return transforms.ToTensor()(low_res_img)

def main(args):
    data_path = args.data_path
    images = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('jpg', 'png', 'jpeg'))]

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, image_path in enumerate(images):
        if idx % 50 == 0:
            print("{} out of {}".format(idx, len(images)))
        print(f"Processing {image_path}...")
        data = process_image(image_path)
        output_path = os.path.join(output_dir, "{}_processed.jpg".format(image_path.split("/")[-1].split('.')[0]))
        print(f"Saving to {output_path}")
        save_image(data, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low Resolution and Low Light Image Simulation Script")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the folder containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the folder to save output images')
    args = parser.parse_args()
    main(args)
