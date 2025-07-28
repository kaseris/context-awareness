import io
import requests
from PIL import Image
import torch
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id

url = "/Users/michaliskaseris/Documents/dev/content-awareness/parrot.jpeg"
image = Image.open(url)

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

# prepare image for the model
inputs = feature_extractor(images=image, return_tensors="pt")

# forward pass
outputs = model(**inputs)

# use the `post_process_panoptic` method of `DetrFeatureExtractor` to convert to COCO format
processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

# the segmentation is stored in a special-format png
panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
# retrieve the ids corresponding to each mask
panoptic_seg_id = rgb_to_id(panoptic_seg)

# Create visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Original image
ax1.imshow(image)
ax1.set_title('Original Image')
ax1.axis('off')

# Segmentation IDs as grayscale
im2 = ax2.imshow(panoptic_seg_id, cmap='tab20', alpha=0.8)
ax2.set_title('Segmentation IDs (Grayscale)')
ax2.axis('off')

# Overlay segmentation on original image
ax3.imshow(image)
overlay = ax3.imshow(panoptic_seg_id, cmap='tab20', alpha=0.6)
ax3.set_title('Segmentation IDs Overlaid on Original Image')
ax3.axis('off')

# Add colorbar to show ID mapping
cbar = plt.colorbar(im2, ax=ax3, shrink=0.8)
cbar.set_label('Segmentation ID')

# Get unique IDs and create legend
unique_ids = numpy.unique(panoptic_seg_id)
colors = plt.cm.tab20(numpy.linspace(0, 1, len(unique_ids)))
patches = [mpatches.Patch(color=colors[i], label=f'ID: {int(unique_ids[i])}') 
           for i in range(len(unique_ids))]

# Add legend
ax3.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Number of unique segmentation IDs: {len(unique_ids)}")
print(f"Segmentation IDs found: {unique_ids}")
print(f"Image shape: {panoptic_seg_id.shape}")

