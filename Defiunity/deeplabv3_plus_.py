import torch
import torch.nn as nn
import torchvision.models as models

class ASPP(nn.Module):
    def __init__(self, in_channels):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, dilation=6, padding=6, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, dilation=12, padding=12, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.aspp5 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, dilation=18, padding=18, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.concat_conv = nn.Sequential(
            nn.Conv2d(256 * 5, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.size()[2:]
        y1 = self.aspp1(x)
        y1 = nn.functional.interpolate(y1, size=size, mode='bilinear', align_corners=False)
        y2 = self.aspp2(x)
        y3 = self.aspp3(x)
        y4 = self.aspp4(x)
        y5 = self.aspp5(x)
        out = torch.cat([y1, y2, y3, y4, y5], dim=1)
        out = self.concat_conv(out)
        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.3),
        )
        self.residual_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x, low_level_features):
        x = torch.cat((x, low_level_features), dim=1)
        x = self.conv1(x)
        residual = x
        x = self.residual_conv(x)
        x += residual
        x = self.relu(x)
        x = self.final_conv(x)
        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int):
        super(DeepLabV3Plus, self).__init__()
        self.backbone = models.resnet101(pretrained=True)
        self.layer0 = nn.Sequential(*list(self.backbone.children())[:3])
        self.layer1 = list(self.backbone.children())[3]
        self.layer2 = list(self.backbone.children())[4]
        self.layer3 = list(self.backbone.children())[5]
        self.layer4 = list(self.backbone.children())[6]

        self.aspp = ASPP(1024)
        self.conv1x1 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.decoder = Decoder(num_classes)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.aspp(x4)
        x = self.upsample(x)

        x1 = self.conv1x1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        # Ensure alignment
        if x.size(2) != x1.size(2) or x.size(3) != x1.size(3):
            x = nn.functional.interpolate(x, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)

        x = self.decoder(x, x1)
        return x

    

# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torch.nn.functional as F

# # Define the ASPP (Atrous Spatial Pyramid Pooling) module
# class ASPP(nn.Module):
#     """
#     Different dilation rates to capture multi-scale features.
#     Preserve spatial details and global context.
#     """
#     def __init__(self, in_channels):
#         super(ASPP, self).__init__()
#         self.aspp1 = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.aspp2 = nn.Sequential(
#             nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.aspp3 = nn.Sequential(
#             nn.Conv2d(in_channels, 256, kernel_size=3, dilation=6, padding=6, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.aspp4 = nn.Sequential(
#             nn.Conv2d(in_channels, 256, kernel_size=3, dilation=12, padding=12, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.aspp5 = nn.Sequential(
#             nn.Conv2d(in_channels, 256, kernel_size=3, dilation=18, padding=18, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.concat_conv = nn.Sequential(
#             nn.Conv2d(256 * 5, 256, kernel_size=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         size = x.size()[2:]
#         y1 = self.aspp1(x)
#         y1 = F.interpolate(y1, size=size, mode='bilinear', align_corners=False)  # Upsample back to original size

#         y2 = self.aspp2(x)
        
#         y3 = self.aspp3(x)
#         y3 = F.interpolate(y3, size=size, mode='bilinear', align_corners=False)
        
#         y4 = self.aspp4(x)
#         y4 = F.interpolate(y4, size=size, mode='bilinear', align_corners=False)
        
#         y5 = self.aspp5(x)
#         y5 = F.interpolate(y5, size=size, mode='bilinear', align_corners=False)
        
#         out = torch.cat([y1, y2, y3, y4, y5], dim=1)
#         out = self.concat_conv(out)
#         return out


# # Define the Decoder
# class Decoder(nn.Module):
#     def __init__(self, num_classes):
#         super(Decoder, self).__init__()
#         # Decoder that refines the output with convolutions
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Dropout2d(0.3),  # Regularization
#         )
#         self.residual_conv = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#         )
#         self.relu = nn.ReLU()
#         self.final_conv = nn.Conv2d(256, num_classes, kernel_size=1)

#     def forward(self, x):
#         # Pass through the convolution and residual layers
#         x = self.conv1(x)
#         residual = x
#         x = self.residual_conv(x)
#         x += residual  # Skip connection for better gradient flow
#         x = self.relu(x)

#         # Final classification layer
#         x = self.final_conv(x)
#         return x


# # Define the DeepLabV3Plus model
# class DeepLabV3Plus(nn.Module):
#     def __init__(self, num_classes: int):
#         super(DeepLabV3Plus, self).__init__()
        
#         # Load ResNet101 as the backbone
#         self.backbone = models.resnet101(pretrained=True)
        
#         # Extract relevant layers
#         self.layer0 = nn.Sequential(*list(self.backbone.children())[:3])
#         self.layer1 = list(self.backbone.children())[3]
#         self.layer2 = list(self.backbone.children())[4]
#         self.layer3 = list(self.backbone.children())[5]
#         self.layer4 = list(self.backbone.children())[6]
        
#         # Freeze backbone layers (optional for fine-tuning)
#         for param in self.backbone.parameters():
#             param.requires_grad = False
        
#         # ASPP module
#         self.aspp = ASPP(1024)
        
#         # Decoder to refine the output
#         self.decoder = Decoder(num_classes)
        
#         # Upsample to original image size
#         self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
    
#     def forward(self, x):
#         # Backbone feature extraction
#         x0 = self.layer0(x)
#         x1 = self.layer1(x0)  # Low-level features
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)  # High-level features
        
#         # Pass high-level features through ASPP
#         x = self.aspp(x4)
        
#         # Upsample high-level features to match low-level features size
#         x = self.upsample(x)
        
#         # Reduce low-level features to 256 channels
#         x1 = self.conv1x1(x1)
#         x1 = self.bn1(x1)
#         x1 = self.relu1(x1)
        
#         # Check if the sizes match, otherwise interpolate
#         if x.size(2) != x1.size(2) or x.size(3) != x1.size(3):
#             x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode='bilinear', align_corners=False)
        
#         # Concatenate high-level (ASPP) and low-level features
#         x = torch.cat((x, x1), dim=1) 
        
#         # Pass through the decoder
#         x = self.decoder(x)
        
#         return x