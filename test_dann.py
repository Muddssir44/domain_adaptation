import torch
import torch.nn as nn
from dann_model import DANN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load the saved model
num_classes = 31  # Replace with the actual number of classes if different
model = DANN(num_classes=num_classes)
model.load_state_dict(torch.load("dann_model_optimized.pth"))
model.eval()  # Set the model to evaluation mode

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()

# Define transformation for test set
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define test data paths
amazon_test_dir = './archive/amazon/test'
webcam_test_dir = './archive/webcam/test'

# Load test datasets
amazon_test_data = datasets.ImageFolder(amazon_test_dir, transform=transform)
webcam_test_data = datasets.ImageFolder(webcam_test_dir, transform=transform)

# Create DataLoaders for test data
amazon_test_loader = DataLoader(amazon_test_data, batch_size=32, shuffle=False)
webcam_test_loader = DataLoader(webcam_test_data, batch_size=32, shuffle=False)

# Evaluation function
def evaluate(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda() if torch.cuda.is_available() else (images, labels)
            outputs, _ = model(images)  # We only need the label output, not the domain output
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

# Evaluate the model on the test sets
print("Evaluating on Amazon test set...")
evaluate(model, amazon_test_loader)

print("Evaluating on Webcam test set...")
evaluate(model, webcam_test_loader)
