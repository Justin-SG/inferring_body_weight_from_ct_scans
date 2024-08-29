import torch
import torch.nn as nn


class CTWeightRegressor2D(nn.Module):
    def __init__(self, backend_model, num_additional_params=0, fc_layers=[128, 64, 32]):
        super(CTWeightRegressor2D, self).__init__()

        # Assume the backend model is already instantiated and passed as an argument
        self.backend = backend_model

        # Determine the feature dimension based on the backend model
        if hasattr(self.backend, 'fc'):  # e.g., ResNet
            self.feature_dim = self.backend.fc.in_features
            self.backend.fc = nn.Identity()  # Remove the original fully connected layer
        elif hasattr(self.backend, 'heads'):  # e.g., Vision Transformer
            self.feature_dim = self.backend.heads.head.in_features
            self.backend.heads.head = nn.Identity()  # Remove the original fully connected layer
        else:
            raise ValueError("Unsupported backend model structure")

        # Total input size for the fully connected layers
        input_dim = self.feature_dim + num_additional_params

        # Create the fully connected layers
        fc_layers_list = []
        in_features = input_dim
        for out_features in fc_layers:
            fc_layers_list.append(nn.Linear(in_features, out_features))
            fc_layers_list.append(nn.ReLU(inplace=True))
            in_features = out_features

        # Final output layer: 1 output for the weight regression
        fc_layers_list.append(nn.Linear(in_features, 1))
        self.fc = nn.Sequential(*fc_layers_list)

    def forward(self, x, additional_params=None):
        # Pass the input through the backend model
        features = self.backend(x)

        # If additional parameters are provided, concatenate them with the features
        if additional_params is not None:
            additional_params = additional_params.to(features.device)  # Ensure same device
            features = torch.cat([features, additional_params], dim=1)

        # Pass the concatenated tensor through the fully connected layers
        output = self.fc(features)
        return output



class ScaledOutputRegressor(nn.Module):
    def __init__(self, backend_model, fc_layers=[128, 64, 32]):
        super(ScaledOutputRegressor, self).__init__()

        # Assume the backend model is already instantiated and passed as an argument
        self.backend = backend_model

        # Determine the feature dimension based on the backend model
        if hasattr(self.backend, 'fc'):  # e.g., ResNet
            self.feature_dim = self.backend.fc.in_features
            self.backend.fc = nn.Identity()  # Remove the original fully connected layer
        elif hasattr(self.backend, 'heads'):  # e.g., Vision Transformer
            self.feature_dim = self.backend.heads.head.in_features
            self.backend.heads.head = nn.Identity()  # Remove the original fully connected layer
        else:
            raise ValueError("Unsupported backend model structure")

        # Create the fully connected layers
        fc_layers_list = []
        in_features = self.feature_dim
        for out_features in fc_layers:
            fc_layers_list.append(nn.Linear(in_features, out_features))
            fc_layers_list.append(nn.ReLU(inplace=True))
            in_features = out_features

        # Final output layer: 1 output for the regression
        fc_layers_list.append(nn.Linear(in_features, 1))
        self.fc = nn.Sequential(*fc_layers_list)

    def forward(self, x, scaling_factor=1.0):
        # Pass the input through the backend model
        features = self.backend(x)

        # Apply the scaling factor to the features
        scaled_features = features * scaling_factor

        # Pass the scaled features through the fully connected layers
        output = self.fc(scaled_features)
        return output