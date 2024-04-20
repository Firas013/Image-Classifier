# Same imports
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

# Import train.py
import train


def predict_parser():
    parser = argparse.ArgumentParser(description='Predict file for predicting the model...')
    
    parser.add_argument('--image', action='store',default = '/aipnd-project/flowers/test/56/image_02825.jpg',
                        description='The image that we want to predicting it')
    parser.add_argument('--checkpoint', action='store', description=' For loading the checkpoint for out model..')
    parser.add_arguemnt('--top_k', type=int, default=5, description='Printing  the top 5 probabilites')
    parser.add_argument('--category_names', type=str, help='Categories the probabilites')
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    
    args = parser.parse_args()
    
    return args


def loading_model(checkpoint_path):
    
    if gpu_mode == True:
        checkpoint = torch.load(save_dir)
        
    model = models.vgg11(pretrained = True)

    for param in model.parameters():
    param.requires_grad = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    model.to(device)
    load_parameters = torch.load(save_dir, map_location=torch.device('cpu'))

    # Set the attribute 
    model.classifier = load_parameters['classifier']
    model.class_to_idx = load_parameters['map']
    model.load_state_dict(load_parameters['state_dict'])
    
    return model

def process_image(image_path):
    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Open and preprocess image
    img = Image.open(image_path)
    img = preprocess(img)
    
    return img


def predict(image_path, model, top_k):
    
    model.to('cpu')
    
    # Pre-process the image
    image_tensor = process_image(image_path)
    # Adding a new dimension to convert a single image tensor into a batch containing one image tensor
    image_tensor = torch.unsqueeze(image_tensor, 0) # Dimension size of 1
    
    # Move model to CPU and set to evaluation mode
    model.eval()
    
    # Disable gradient calculation
    with torch.no_grad():
        
        # Feed image through the model
        output = model(image_tensor)
    
    # Get top k probabilities and class indices
    probabilities = torch.exp(output)
    predicts_topk = probabilities.topk(top_k)[0]  # Top k probabilities
    indices_topk = probabilities.topk(top_k)[1]   # Indices of top k probabilities
    
    # Tensors to numpy arrays
    probabilities = predicts_topk.squeeze().numpy()
    indices = indices_topk.squeeze().numpy()
    
    # Map indices to class labels and probabilities to class names
    idx_to_class = {idx: class_ for class_, idx in model.class_to_idx.items()}
    labels = [idx_to_class[idx] for idx in indices]
    flowers = [cat_to_name[label] for label in labels]
    
    return probabilities.tolist(), labels, flowers



def main():
    # Parse command-line arguments
    args = predict_parser()

    # Load the mapping of categories to names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load the trained model from checkpoint
    model = loading_model(args.checkpoint)

    # Process the input image
    image_tensor = process_image(args.image)

    # Determine the device (CPU or GPU)
    device = torch.device("cuda" if args.gpu == "gpu" and torch.cuda.is_available() else "cpu")

    # Perform prediction
    top_probs, top_labels, top_flowers = predict(args.image, model, device, args.top_k)


if __name__ == '__main__':
    main()

