import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Define the DANN model
class DANN(nn.Module):
    def __init__(self, num_classes=31):
        super(DANN, self).__init__()

        # Feature extractor: Use pre-trained ResNet-50
        self.feature_extractor = models.resnet50(weights='IMAGENET1K_V1')
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()  # Remove the original classifier layer

        # Label classifier
        self.label_classifier = nn.Sequential(
            nn.Linear(num_ftrs, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(num_ftrs, 100),
            nn.ReLU(),
            nn.Linear(100, 2)  # Binary classification: Amazon or Webcam
        )

    def forward(self, x, alpha=1.0):
        # Extract features
        features = self.feature_extractor(x)

        # Label prediction
        label_output = self.label_classifier(features)

        # Gradient reversal for domain adaptation
        reverse_features = GradReverse.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return label_output, domain_output

# Gradient reversal layer for adversarial training
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
