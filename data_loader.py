import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define paths to your data
data_dir = './archive'  # Update this if your 'amazon' and 'webcam' folders are located elsewhere
amazon_train_dir = os.path.join(data_dir, 'amazon', 'train')
amazon_val_dir = os.path.join(data_dir, 'amazon', 'val')
webcam_train_dir = os.path.join(data_dir, 'webcam', 'train')
webcam_val_dir = os.path.join(data_dir, 'webcam', 'val')

# Define transformations for the dataset
# These will resize the images, convert them to tensors, and normalize them
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets for each domain
amazon_train_data = datasets.ImageFolder(amazon_train_dir, transform=transform)
amazon_val_data = datasets.ImageFolder(amazon_val_dir, transform=transform)
webcam_train_data = datasets.ImageFolder(webcam_train_dir, transform=transform)
webcam_val_data = datasets.ImageFolder(webcam_val_dir, transform=transform)

# Create DataLoaders to load the data in batches
batch_size = 32
amazon_train_loader = DataLoader(amazon_train_data, batch_size=batch_size, shuffle=True)
amazon_val_loader = DataLoader(amazon_val_data, batch_size=batch_size, shuffle=False)
webcam_train_loader = DataLoader(webcam_train_data, batch_size=batch_size, shuffle=True)
webcam_val_loader = DataLoader(webcam_val_data, batch_size=batch_size, shuffle=False)

# Verify that data loading works by printing the shape of one batch
# This code prints the shape of images and labels from one batch in each domain
def check_data_loaders():
    for images, labels in amazon_train_loader:
        print("Amazon - Train Batch - images:", images.shape, "labels:", labels.shape)
        break

    for images, labels in amazon_val_loader:
        print("Amazon - Validation Batch - images:", images.shape, "labels:", labels.shape)
        break

    for images, labels in webcam_train_loader:
        print("Webcam - Train Batch - images:", images.shape, "labels:", labels.shape)
        break

    for images, labels in webcam_val_loader:
        print("Webcam - Validation Batch - images:", images.shape, "labels:", labels.shape)
        break

# Run the check function to verify everything works
if __name__ == "__main__":
    check_data_loaders()
