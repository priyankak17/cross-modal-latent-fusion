from torchvision import transforms
from torchvision.utils import save_image
import torch
import os
import cv2
import numpy as np
import argparse

"""
NOTE!: Must have torch==0.4.1 and torchvision==0.2.1
The sketch simplification model (sketch_gan.t7) from Simo Serra et al. can be downloaded from their official implementation: 
    https://github.com/bobbens/sketch_simplification
"""

def sobel(img):
    opImgx = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
    opImgy = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    return cv2.bitwise_or(opImgx, opImgy)

def sketch(frame):
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    invImg = 255 - frame
    edgImg0 = sobel(frame)
    edgImg1 = sobel(invImg)
    edgImg = cv2.addWeighted(edgImg0, 0.75, edgImg1, 0.75, 0)
    opImg = 255 - edgImg
    return opImg

def get_sketch_image(image_path):
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    sketch_image = sketch(original)
    return sketch_image[:, :, np.newaxis]

def main(args):
    use_cuda = True

    # Load the model using torch.load
    checkpoint = torch.load(args.model_path)
    model = checkpoint['model_gan_sketch.pth']  # Adjust according to the actual structure of the checkpoint
    immean = torch.tensor(checkpoint['mean'], dtype=torch.float32)
    imstd = torch.tensor(checkpoint['std'], dtype=torch.float32)
    model.eval()

    data_path = args.data_path
    images = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('.jpg', '.png'))]

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, image_path in enumerate(images):
        if idx % 50 == 0:
            print("{} out of {}".format(idx, len(images)))
        data = get_sketch_image(image_path)
        data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)
        if use_cuda:
            model = model.cuda()
            data = data.cuda()
        with torch.no_grad():
            pred = model(data).float()
        save_image(pred[0], os.path.join(output_dir, "{}_edges.jpg".format(image_path.split("/")[-1].split('.')[0])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sketch Generation Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model .pth file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the folder containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the folder to save output images')
    args = parser.parse_args()
    main(args)
