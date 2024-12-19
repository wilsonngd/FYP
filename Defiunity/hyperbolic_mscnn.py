import torch
import torch.nn as nn
from hypll import nn as hnn
from hypll.tensors import ManifoldTensor
from hypll.manifolds.poincare_ball import PoincareBall, Curvature
from hypll.tensors import TangentTensor

# Define the manifold
manifold = PoincareBall(c=Curvature(requires_grad=True))

# class HyperbolicResidualBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         manifold: PoincareBall,
#         stride: int = 1,
#         downsample: Optional[nn.Sequential] = None,
#     ):
#         super(HyperbolicResidualBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.manifold = manifold
#         self.downsample = downsample

#         self.conv1 = hnn.HConvolution2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             manifold=manifold,
#             stride=1,
#             padding=1
#         )
#         self.bn1 = hnn.HBatchNorm2d(features=out_channels, manifold=manifold)
#         self.relu = hnn.HReLU(manifold=manifold)
#         self.conv2 = hnn.HConvolution2d(
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             manifold=manifold,
#             padding=1,
#         )
#         self.bn2 = hnn.HBatchNorm2d(features=out_channels, manifold=manifold)

#     def forward(self, x: ManifoldTensor) -> ManifoldTensor:
#         residual = x
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)

#         if self.downsample is not None:
#             residual = self.downsample(residual)

#         x = self.manifold.mobius_add(x, residual)
#         x = self.relu(x)

#         return x

class HyperbolicMSCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super(HyperbolicMSCNN, self).__init__()
        self.manifold = manifold

        # Define multiple scales with hyperbolic layers
        self.scale1 = nn.Sequential(
            hnn.HConvolution2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1, manifold=self.manifold),
            hnn.HReLU(manifold=self.manifold),
            hnn.HMaxPool2d(kernel_size=(2, 2), manifold=self.manifold)
        )

        self.scale2 = nn.Sequential(
            hnn.HConvolution2d(in_channels=32, out_channels=64, kernel_size=5, padding=2, manifold=self.manifold),
            hnn.HReLU(manifold=self.manifold),
            hnn.HMaxPool2d(kernel_size=(2, 2), manifold=self.manifold)
        )

        self.scale3 = nn.Sequential(
            hnn.HConvolution2d(in_channels=64, out_channels=128, kernel_size=7, padding=3, manifold=self.manifold),
            hnn.HReLU(manifold=self.manifold),
            hnn.HMaxPool2d(kernel_size=(2, 2), manifold=self.manifold)
        )

        self.scale4 = nn.Sequential(
            hnn.HConvolution2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, manifold=self.manifold),
            hnn.HReLU(manifold=self.manifold),
            hnn.HMaxPool2d(kernel_size=(2, 2), manifold=self.manifold)
        )

        # Define global average pooling and fully connected layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = hnn.HLinear(in_features=(32 + 64 + 128 + 128), out_features=512, manifold=self.manifold)
        self.relu = hnn.HReLU(manifold=self.manifold)
        self.fc2 = hnn.HLinear(in_features=512, out_features=num_classes, manifold=self.manifold)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        try:
            # print(f"Input to model: {x.shape}")
            x1 = self.scale1(x)
            # print(f"After scale1: {x1.shape}")

            x2 = self.scale2(x1)
            # print(f"After scale2: {x2.shape}")

            x3 = self.scale3(x2)
            # print(f"After scale3: {x3.shape}")

            x4 = self.scale4(x3)
            # print(f"After scale4: {x4.shape}")

            # Use adaptive pooling to bring all feature maps to the same size
            target_size = (x4.shape[2], x4.shape[3])  # Use the size from the last scale

            x1_pooled = nn.functional.adaptive_avg_pool2d(x1.tensor, output_size=target_size)
            x2_pooled = nn.functional.adaptive_avg_pool2d(x2.tensor, output_size=target_size)
            x3_pooled = nn.functional.adaptive_avg_pool2d(x3.tensor, output_size=target_size)

            # Concatenate features from all scales
            concatenated = torch.cat([x1_pooled, x2_pooled, x3_pooled, x4.tensor], dim=1)
            # print(f"After concatenation: {concatenated.shape}")

            # Wrap the concatenated tensor back into a ManifoldTensor
            concatenated_manifold = ManifoldTensor(concatenated, manifold=self.manifold)
            # print(f"Passed: {concatenated_manifold.shape}")

            # Apply global average pooling
            pooled = self.global_avg_pool(concatenated_manifold.tensor)
            # print(f"After global_avg_pool: {pooled.shape}")

            # Flatten the pooled features
            flattened = torch.flatten(pooled, 1)
            # print(f"After flattening: {flattened.shape}")

            flattened_manifold = ManifoldTensor(flattened, manifold=self.manifold)

            # Pass through fully connected layers
            x = self.fc1(flattened_manifold)
            # print(f"After self.fc1")
            x = self.relu(x)
            # print(f"After self.relu")
            x = self.fc2(x)
            # print(f"Output from model: {x.shape}")

            return x
        except Exception as e:
            print(f"Error during forward pass: {e}")
            raise e
