import torch
import torch.nn as nn
import torch.nn.functional as F

class MSCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super(MSCNN, self).__init__()

        # Define multiple scales with additional convolutional layers
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.scale4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Define global average pooling and fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=(32 + 64 + 128 + 128), out_features=512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # Process input through each scale
            x1 = self.scale1(x)
            x2 = self.scale2(x1)
            x3 = self.scale3(x2)
            x4 = self.scale4(x3)

            # Use adaptive pooling to bring all feature maps to the same size
            target_size = (x4.shape[2], x4.shape[3])  # Use the size from the last scale
            x1_pooled = F.adaptive_avg_pool2d(x1, output_size=target_size)
            x2_pooled = F.adaptive_avg_pool2d(x2, output_size=target_size)
            x3_pooled = F.adaptive_avg_pool2d(x3, output_size=target_size)

            # Concatenate features from all scales
            concatenated = torch.cat([x1_pooled, x2_pooled, x3_pooled, x4], dim=1)

            # Apply global average pooling
            pooled = self.global_avg_pool(concatenated)

            # Flatten the pooled features
            flattened = torch.flatten(pooled, 1)

            # Pass through fully connected layers
            x = self.fc1(flattened)
            x = self.relu(x)
            x = self.fc2(x)

            return x
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise e
