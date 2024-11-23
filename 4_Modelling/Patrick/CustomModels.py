import torch
import torch.nn as nn


class ResNetRegression(nn.Module):
    def __init__(self, pretrained_model, num_outputs=1):
        super(ResNetRegression, self).__init__()
        
        # Keep all layers of the pretrained model
        self.backbone = nn.Sequential(*list(pretrained_model.children())[:-1])
        
        # Add global average pooling (if not already included in your pretrained model)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        # Infer the feature size of the backbone dynamically
        with torch.no_grad():
            # Pass a dummy input through the backbone to get the feature size
            dummy_input = torch.zeros(3, 1, 112, 224, 224)  # Adjust dimensions if necessary for 3D or specific models
            feature_size = self.backbone(dummy_input).shape[1]
        
        # Add the regression head (a fully connected layer)
        self.fc = nn.Linear(feature_size, num_outputs)  # 2048 is the output of the last conv layer
        
    def forward(self, x):
        # Forward pass through the backbone
        x = self.backbone(x)
        
        # Global average pooling to reduce the 3D output to a single vector
        x = self.global_pool(x)
        
        # Flatten the pooled output
        x = torch.flatten(x, 1)
        
        # Regression output
        x = self.fc(x)
        
        return x
        