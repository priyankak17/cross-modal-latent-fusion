import os
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Extract descriptors from low-resolution images")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing low-res images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for descriptor .pth files")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--workers", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--model_path", type=str, help="Path to custom pre-trained model")
    return parser.parse_args()

def load_model(model_path=None):
    if model_path and os.path.exists(model_path):
        print(f"Loading custom model from {model_path}")
        model = torch.load(model_path, map_location='cuda')
    else:
        print("Using default ResNet50 model")
        model = resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove the last FC layer
    
    model.eval()
    return model.cuda()

def process_images(model, image_paths, transform):
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img)
        images.append(img)
    images = torch.stack(images).cuda()
    with torch.no_grad():
        features = model(images)
    return features.squeeze()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load pre-trained model
    model = load_model(args.model_path)
    
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get all image files
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Process images in batches
    for i in tqdm(range(0, len(image_files), args.batch_size)):
        batch_files = image_files[i:i+args.batch_size]
        batch_paths = [os.path.join(args.input_dir, f) for f in batch_files]
        
        features = process_images(model, batch_paths, transform)
        
        # Save descriptors
        for j, file in enumerate(batch_files):
            descriptor = features[j]
            output_path = os.path.join(args.output_dir, f"{os.path.splitext(file)[0]}_descriptor.pth")
            torch.save(descriptor, output_path)

    print(f"Descriptors extracted and saved in {args.output_dir}")

if __name__ == "__main__":
    main()