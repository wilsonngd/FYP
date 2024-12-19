import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
import torch

# Define the transformation for the segmentation model
class SegmentTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
    def __call__(self, image):
        return self.transform(image)

# Define function for the segmentation and cropping
def process_and_crop_segmented_image(model, image, device, target_size=(256, 256), mask_color=(255, 255, 255), exclude_classes=None):
    # Preprocess the image
    transform = SegmentTransform()
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Predict segmentation mask
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        output_resized = torch.nn.functional.interpolate(output, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        pred_mask = torch.argmax(torch.nn.functional.softmax(output_resized, dim=1), dim=1)[0].cpu().numpy()

    # Resize the predicted mask to match the original image size
    pred_mask_resized = np.array(
        Image.fromarray(pred_mask.astype(np.uint8)).resize(image.size, resample=Image.NEAREST)
    )

    # Exclude specific classes by zeroing out those regions in the mask
    if exclude_classes is not None:
        for cls in exclude_classes:
            pred_mask_resized[pred_mask_resized == cls] = 0

    # Crop the segmented area based on the filtered mask
    non_zero_indices = np.argwhere(pred_mask_resized > 0)
    if non_zero_indices.size == 0:
        raise ValueError("No segmented area found after excluding specified classes.")

    y_min, x_min = non_zero_indices.min(axis=0)
    y_max, x_max = non_zero_indices.max(axis=0)
    cropped_image = image.crop((x_min, y_min, x_max + 1, y_max + 1))

    # Resize and pad the cropped image to the target size
    cropped_image = ImageOps.pad(cropped_image, target_size, color=mask_color)

    return cropped_image