import torch
import torch.optim as optim
import torch.nn as nn
from dann_model import DANN
from data_loader import amazon_train_loader, amazon_val_loader, webcam_train_loader, webcam_val_loader

# Initialize the DANN model
num_classes = len(amazon_train_loader.dataset.classes)  # Get the number of classes from data
model = DANN(num_classes=num_classes).cuda() if torch.cuda.is_available() else DANN(num_classes=num_classes)

# Loss functions and optimizer
criterion_label = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate

# Add a learning rate scheduler for dynamic adjustment
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduces LR by half every 5 epochs

# Training function
def train_dann(model, source_loader, target_loader, epochs=20, alpha=0.1):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):
            source_data, source_labels = source_data.cuda(), source_labels.cuda() if torch.cuda.is_available() else (source_data, source_labels)
            target_data = target_data.cuda() if torch.cuda.is_available() else target_data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Source domain - Train with class labels
            label_output, domain_output = model(source_data, alpha=alpha)
            label_loss = criterion_label(label_output, source_labels)
            domain_loss_source = criterion_domain(domain_output, torch.zeros(source_data.size(0)).long().cuda() if torch.cuda.is_available() else torch.zeros(source_data.size(0)).long())

            # Target domain - Train without class labels (domain adaptation only)
            _, domain_output = model(target_data, alpha=alpha)
            domain_loss_target = criterion_domain(domain_output, torch.ones(target_data.size(0)).long().cuda() if torch.cuda.is_available() else torch.ones(target_data.size(0)).long())

            # Combine losses
            loss = label_loss + domain_loss_source + domain_loss_target
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Step the learning rate scheduler
        scheduler.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}")

    print("Training complete.")

# Evaluation function
def evaluate(model, test_loader):
    model.eval()  # Set the model to evaluation mode
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
    print(f"Accuracy on test data: {accuracy:.2f}%")

# Load test data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformation for training set with augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define transformation for test set (no augmentation)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define test data paths
amazon_test_dir = './archive/amazon/test'
webcam_test_dir = './archive/webcam/test'

# Load test datasets
amazon_test_data = datasets.ImageFolder(amazon_test_dir, transform=test_transform)
webcam_test_data = datasets.ImageFolder(webcam_test_dir, transform=test_transform)

# Create DataLoaders for test data
amazon_test_loader = DataLoader(amazon_test_data, batch_size=32, shuffle=False)
webcam_test_loader = DataLoader(webcam_test_data, batch_size=32, shuffle=False)

# Run training with adjusted parameters
train_dann(model, amazon_train_loader, webcam_train_loader, epochs=20, alpha=0.1)

# Save the trained model
torch.save(model.state_dict(), "dann_model_optimized.pth")
print("Model saved as dann_model_optimized.pth")

# Evaluate the model on the test sets
print("Evaluating on Amazon test set...")
evaluate(model, amazon_test_loader)

print("Evaluating on Webcam test set...")
evaluate(model, webcam_test_loader)
