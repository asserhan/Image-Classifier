import argparse
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models

def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Classifier Training')
    parser.add_argument('data_dir', help='Directory containing the dataset')
    parser.add_argument('--save_dir', default="checkpoint.pth", help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=150, help='Number of hidden units')
    parser.add_argument('--output_features', type=int, default=102, help='Number of output features')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--arch', default='resnet18', help='Model architecture')
    return parser.parse_args()

def create_data_transforms():
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, valid_transform

def load_datasets(data_dir, train_transform, valid_transform):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_set = datasets.ImageFolder(valid_dir, transform=valid_transform)
    
    return train_set, valid_set

def create_dataloaders(train_set, valid_set, batch_size=64):
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
    return trainloader, validloader

def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

def load_pretrained_model(arch):
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model

def create_classifier(model, hidden_units, output_features):
    if hasattr(model, 'classifier'):
        in_features = model.classifier.in_features
    else:
        in_features = model.fc.in_features

    return nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, output_features),
        nn.LogSoftmax(dim=1)
    )

def train_model(model, trainloader, validloader, device, optimizer, criterion, epochs=5, print_every=10):
    steps = 0
    for epoch in range(epochs):
        running_loss = 0
        model.train()
        
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validation_loss, accuracy = validate_model(model, validloader, criterion, device)
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss:.3f}.. "
                      f"Validation accuracy: {accuracy:.3f}")
                running_loss = 0
                model.train()

    return model

def validate_model(model, validloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            test_loss += loss.item()

            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return test_loss / len(validloader), accuracy / len(validloader)

def save_checkpoint(model, optimizer, class_to_idx, path, arch, hidden_units, output_features):
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'arch': arch,
        'hidden_units': hidden_units,
        'output_features': output_features
    }
    torch.save(checkpoint, path)

def main():
    args = parse_arguments()

    train_transform, valid_transform = create_data_transforms()
    train_set, valid_set = load_datasets(args.data_dir, train_transform, valid_transform)
    trainloader, validloader = create_dataloaders(train_set, valid_set)

    device = get_device(args.gpu)
    model = load_pretrained_model(args.arch)

    if hasattr(model, 'classifier'):
        model.classifier = create_classifier(model, args.hidden_units, args.output_features)
    else:
        model.fc = create_classifier(model, args.hidden_units, args.output_features)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    model.to(device)

    trained_model = train_model(model, trainloader, validloader, device, optimizer, criterion, args.epochs)
    save_checkpoint(trained_model, optimizer, train_set.class_to_idx, args.save_dir, args.arch, args.hidden_units, args.output_features)

if __name__ == '__main__':
    main()