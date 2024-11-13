import argparse
import json
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Classifier Prediction')
    parser.add_argument('image_dir', help='Path to the image file')
    parser.add_argument('checkpoint_dir', help='Path to the model checkpoint')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for prediction')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', default='cat_to_name.json', help='Path to category to name mapping JSON')
    return parser.parse_args()

def load_pretrained_model(arch):
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model

def create_classifier(model, hidden_units=150, out_features=102):
    in_features = model.fc.in_features
    return nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, out_features),
        nn.LogSoftmax(dim=1)
    )

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    output_features = checkpoint['output_features']

    model = load_pretrained_model(arch)
    if hasattr(model, 'classifier'):
        model.classifier = create_classifier(model, hidden_units, output_features)
    else:
        model.fc = create_classifier(model, hidden_units, output_features)

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    image_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with Image.open(image_path) as img:
        return image_transforms(img)

def predict(image_path, model, topk, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        input_img = process_image(image_path).unsqueeze(0).to(device)
        output = model(input_img)
        probabilities = F.softmax(output, dim=1)
        top_probs, top_indices = probabilities.topk(topk)

    return top_probs.cpu().numpy().squeeze(), top_indices.cpu().numpy().squeeze()

def load_category_names(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def main():
    args = parse_arguments()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint_dir)
    cat_to_name = load_category_names(args.category_names)

    probs, indices = predict(args.image_dir, model, args.top_k, device)

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    labels = [cat_to_name[idx_to_class[idx]] for idx in indices]

    for label, prob in zip(labels, probs):
        print(f"{label} with a probability of {prob:.4f}")

if __name__ == '__main__':
    main()