import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import torch
import numpy as np
from PIL import Image
import io

from transformers import EomtForUniversalSegmentation, AutoImageProcessor
from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id

# Load the same image for both models
image_path = "/Users/michaliskaseris/Documents/dev/content-awareness/parrot.jpeg"
image = Image.open(image_path)

# ===== EOMT Model =====
print("Loading EOMT model...")
eomt_model_id = "tue-mps/coco_panoptic_eomt_large_640"
eomt_processor = AutoImageProcessor.from_pretrained(eomt_model_id)
eomt_model = EomtForUniversalSegmentation.from_pretrained(eomt_model_id)

eomt_inputs = eomt_processor(
    images=image,
    return_tensors="pt",
)

with torch.inference_mode():
    eomt_outputs = eomt_model(**eomt_inputs)

# Post-process EOMT outputs
eomt_target_sizes = [(image.height, image.width)]
eomt_preds = eomt_processor.post_process_panoptic_segmentation(
    eomt_outputs,
    target_sizes=eomt_target_sizes,
)

eomt_segmentation = eomt_preds[0]["segmentation"]

# ===== DETR Model =====
print("Loading DETR model...")
detr_feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
detr_model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

# Prepare image for DETR
detr_inputs = detr_feature_extractor(images=image, return_tensors="pt")

# Forward pass
detr_outputs = detr_model(**detr_inputs)

# Post-process DETR outputs with target size matching original image
detr_target_sizes = [(image.height, image.width)]
detr_result = detr_feature_extractor.post_process_panoptic(detr_outputs, detr_target_sizes)[0]

# Convert DETR segmentation to numpy array
detr_panoptic_seg = Image.open(io.BytesIO(detr_result["png_string"]))
detr_panoptic_seg = np.array(detr_panoptic_seg, dtype=np.uint8)
detr_segmentation = rgb_to_id(detr_panoptic_seg)

# ===== Visualization =====
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('DETR vs EOMT Panoptic Segmentation Comparison', fontsize=16, fontweight='bold')

# Row 1: Original and Individual Results
# Original Image
axes[0, 0].imshow(image)
axes[0, 0].set_title('Original Image', fontweight='bold')
axes[0, 0].axis('off')

# DETR Segmentation
im1 = axes[0, 1].imshow(detr_segmentation, cmap='tab20')
axes[0, 1].set_title('DETR Segmentation IDs', fontweight='bold')
axes[0, 1].axis('off')

# EOMT Segmentation
im2 = axes[0, 2].imshow(eomt_segmentation, cmap='tab20')
axes[0, 2].set_title('EOMT Segmentation IDs', fontweight='bold')
axes[0, 2].axis('off')

# Empty subplot for spacing
axes[0, 3].axis('off')

# Row 2: Overlaid Results
# DETR Overlay
axes[1, 0].imshow(image)
axes[1, 0].imshow(detr_segmentation, cmap='tab20', alpha=0.6)
axes[1, 0].set_title('DETR Overlay', fontweight='bold')
axes[1, 0].axis('off')

# EOMT Overlay
axes[1, 1].imshow(image)
axes[1, 1].imshow(eomt_segmentation, cmap='tab20', alpha=0.6)
axes[1, 1].set_title('EOMT Overlay', fontweight='bold')
axes[1, 1].axis('off')

# Combined comparison
axes[1, 2].imshow(image)
axes[1, 2].imshow(detr_segmentation, cmap='tab20', alpha=0.4)
axes[1, 2].imshow(eomt_segmentation, cmap='tab10', alpha=0.4)
axes[1, 2].set_title('Combined Overlay\n(DETR + EOMT)', fontweight='bold')
axes[1, 2].axis('off')

# Colorbar
cbar = plt.colorbar(im1, ax=axes[1, 3], shrink=0.8)
cbar.set_label('Segmentation ID')
axes[1, 3].axis('off')

plt.tight_layout()
plt.show()

# ===== Statistics =====
print("\n" + "="*50)
print("COMPARISON STATISTICS")
print("="*50)

# DETR stats
detr_unique_ids = np.unique(detr_segmentation)
detr_unique_ids = detr_unique_ids[detr_unique_ids != 0]  # Remove background
print(f"DETR Model:")
print(f"  - Number of segments: {len(detr_unique_ids)}")
print(f"  - Segment IDs: {detr_unique_ids}")
print(f"  - Image shape: {detr_segmentation.shape}")

# EOMT stats
eomt_unique_ids = np.unique(eomt_segmentation)
eomt_unique_ids = eomt_unique_ids[eomt_unique_ids != 0]  # Remove background
print(f"\nEOMT Model:")
print(f"  - Number of segments: {len(eomt_unique_ids)}")
print(f"  - Segment IDs: {eomt_unique_ids}")
print(f"  - Image shape: {eomt_segmentation.shape}")

# Comparison
print(f"\nComparison:")
print(f"  - DETR segments: {len(detr_unique_ids)}")
print(f"  - EOMT segments: {len(eomt_unique_ids)}")
print(f"  - Difference: {abs(len(detr_unique_ids) - len(eomt_unique_ids))}")

# Model information
print(f"\nModel Information:")
print(f"  - DETR: facebook/detr-resnet-50-panoptic")
print(f"  - EOMT: {eomt_model_id}")
