import argparse
from torchvision import transforms
from torchvision.utils import save_image
import torchfile
import os
import cv2
import numpy as np

def sobel(img):
    opImgx = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
    opImgy = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
    return cv2.bitwise_or(opImgx, opImgy)

def sketch(frame):
    frame = cv2.GaussianBlur(frame, (3,3), 0)
    invImg = 255 - frame
    edgImg0 = sobel(frame)
    edgImg1 = sobel(invImg)
    edgImg = cv2.addWeighted(edgImg0, 0.75, edgImg1, 0.75, 0)
    opImg = 255-  edgImg
    return opImg

def get_sketch_image(image_path):
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    sketch_image = sketch(original)
    return sketch_image[:, :, np.newaxis]

def main(args):
    use_cuda = True

    # cache = torchfile.load(args.model_path)
    # print(cache)
    # model = cache['model']
    # immean = cache['mean']
    # imstd = cache['std']
    # model.evaluate()

    data_path = args.data_path
    images = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(('jpg', 'png', 'jpeg'))]

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, image_path in enumerate(images):
        if idx % 50 == 0:
            print("{} out of {}".format(idx, len(images)))
        print(f"Processing {image_path}...")
        data = get_sketch_image(image_path)
        print(f"Sketch image shape: {data.shape}")
        data = ((transforms.ToTensor()(data))).unsqueeze(0)
        print(f"Transformed data shape: {data.shape}")
        # if use_cuda:
        #     pred = model.cuda().forward(data.cuda()).float()
        # else:
        #     pred = model.forward(data)
        pred = data
        output_path = os.path.join(output_dir, "{}_edges.jpg".format(image_path.split("/")[-1].split('.')[0]))
        print(f"Saving to {output_path}")
        #data = data.unsqueeze()
        save_image(pred[0], output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sketch Generation Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model .t7 file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the folder containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the folder to save output images')
    args = parser.parse_args()
    main(args)
