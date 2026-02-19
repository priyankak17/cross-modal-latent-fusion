import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2



def put_text_with_outline(img, text, position, font_scale=1, thickness=2, outline_thickness=3):

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Draw the white outline
    cv2.putText(img, text, position, font, font_scale, (0, 0, 255), outline_thickness, cv2.LINE_AA)
    
    # Draw the black fill
    cv2.putText(img, text, position, font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    
    return img




encoder_inference = os.listdir("normal_latent_output/inference_results")
sketch_inference = os.listdir("sketch_latent_output/inference_results")
mixed_inference = os.listdir("mixed_results")

for i in range(len(encoder_inference)):
    encoder_image_file = os.path.join("normal_latent_output/inference_results", encoder_inference[i])
    sketch_image_file = os.path.join("sketch_latent_output/inference_results", sketch_inference[i])
    mixed_image_file = os.path.join("mixed_results", mixed_inference[i])
    
    
    position = (50, 150)
    font_scale = 7
    thickness = 6
    outline_thickness = 6

    
    
    # read image with numpy
    encoder_image = np.array(Image.open(encoder_image_file))
    # convert to cv2 image
    encoder_image = cv2.cvtColor(encoder_image, cv2.COLOR_BGR2RGB)
    # add text to image with black color and white outline
    img_with_text = put_text_with_outline(encoder_image, 'encoder_image', position, font_scale, thickness, outline_thickness)
    # convert to numpy back
    encoder_image = cv2.cvtColor(encoder_image, cv2.COLOR_RGB2BGR)
    
    # do the same for sketch image
    sketch_image = np.array(Image.open(sketch_image_file))
    sketch_image = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2RGB)
    img_with_text = put_text_with_outline(sketch_image, 'Sketch Image', position, font_scale, thickness, outline_thickness)

    # cv2.putText(sketch_image, '', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    sketch_image = cv2.cvtColor(sketch_image, cv2.COLOR_RGB2BGR)
    
    # do the same for mixed image
    mixed_image = np.array(Image.open(mixed_image_file))
    mixed_image = cv2.cvtColor(mixed_image, cv2.COLOR_BGR2RGB)
    img_with_text = put_text_with_outline(mixed_image, 'mix Image', position, font_scale, thickness, outline_thickness)

    # cv2.putText(mixed_image, 'Mixed Image', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    mixed_image = cv2.cvtColor(mixed_image, cv2.COLOR_RGB2BGR)

    
    
    
    # append all images horizontally
    images = np.concatenate((encoder_image, sketch_image, mixed_image), axis=1)
    
    # save image as test_image
    Image.fromarray(images).save(f'final_visualization/test_image{i}.png')
    



