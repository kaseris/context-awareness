import os
import torch
import numpy as np
from PIL import Image
from transformers import EomtForUniversalSegmentation, AutoImageProcessor

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")



def create_segment_mask(segmentation, target_id, image_shape):
    """Create a binary mask for a specific segment ID"""
    # Convert tensor to numpy if needed
    if hasattr(segmentation, 'cpu'):
        segmentation = segmentation.cpu().numpy()
    elif hasattr(segmentation, 'numpy'):
        segmentation = segmentation.numpy()
    mask = (segmentation == target_id).astype(np.uint8)
    return mask

def extract_segment_image(original_image, segmentation, segment_id):
    """Extract a specific segment from the original image"""
    # Create binary mask for this segment
    mask = create_segment_mask(segmentation, segment_id, original_image.size[::-1])
    
    # Convert original image to numpy array
    img_array = np.array(original_image)
    
    # Create output image with same dimensions as original
    output_image = np.zeros_like(img_array)
    
    # Copy pixels where mask is 1 (segment area)
    output_image[mask == 1] = img_array[mask == 1]
    
    return Image.fromarray(output_image)

def main():
    # Get the best available device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the image
    image_path = "/Users/michaliskaseris/Documents/dev/content-awareness/Cat.jpg"
    image = Image.open(image_path)
    print(f"Original image size: {image.size}")
    
    # Load EOMT model
    print("Loading EOMT model...")
    model_id = "tue-mps/coco_panoptic_eomt_large_640"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = EomtForUniversalSegmentation.from_pretrained(model_id)
    
    # Move model to the best available device
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Process image
    inputs = processor(
        images=image,
        return_tensors="pt",
    )
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    with torch.inference_mode():
        outputs = model(**inputs)
    
    # Post-process outputs
    target_sizes = [(image.height, image.width)]
    predictions = processor.post_process_panoptic_segmentation(
        outputs,
        target_sizes=target_sizes,
    )
    
    segmentation = predictions[0]["segmentation"]
    segments_info = predictions[0]["segments_info"]
    
    # Convert tensor to numpy if needed
    if hasattr(segmentation, 'cpu'):
        segmentation = segmentation.cpu().numpy()
    elif hasattr(segmentation, 'numpy'):
        segmentation = segmentation.numpy()
    
    # Get unique segment IDs from segments_info
    unique_segments = segments_info
    unique_ids = [segment['id'] for segment in unique_segments]
    
    print(f"\nFound {len(unique_segments)} segments:")
    for segment in unique_segments:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        print(f"  ID {segment_id}: {segment_label} (label_id: {segment_label_id})")
    
    # Create output directory
    output_dir = "segmented_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract and save each segment
    print(f"\nExtracting segments to '{output_dir}' directory...")
    
    # Save all segments
    print(f"Saving all {len(unique_segments)} segments...")
    for segment in unique_segments:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        segment_image = extract_segment_image(image, segmentation, segment_id)
        
        filename = f"{segment_label}_id_{segment_id}.png"
        filepath = os.path.join(output_dir, filename)
        segment_image.save(filepath)
        print(f"  Saved: {filename}")
    
    print(f"\n✅ All segments saved to '{output_dir}' directory!")
    print(f"📊 Summary:")
    print(f"   - Original image: {image.size[0]}x{image.size[1]} pixels")
    print(f"   - Segments found: {len(unique_segments)}")
    print(f"   - Files created: {len(unique_segments)}")
    
    # Print segment statistics
    print(f"\n📋 Segment Details:")
    total_pixels = image.size[0] * image.size[1]
    total_segment_pixels = 0
    
    for segment in unique_segments:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = model.config.id2label[segment_label_id]
        mask = create_segment_mask(segmentation, segment_id, image.size[::-1])
        pixel_count = np.sum(mask)
        total_segment_pixels += pixel_count
        percentage = (pixel_count / total_pixels) * 100
        print(f"   - {segment_label} (ID: {segment_id}, label_id: {segment_label_id}): {pixel_count:,} pixels ({percentage:.1f}%)")
    
    # Calculate background pixels
    background_pixels = total_pixels - total_segment_pixels
    background_percentage = (background_pixels / total_pixels) * 100
    
    print(f"   - Background (unassigned): {background_pixels:,} pixels ({background_percentage:.1f}%)")
    print(f"\n📊 Total Coverage: {total_segment_pixels:,} / {total_pixels:,} pixels ({total_segment_pixels/total_pixels*100:.1f}% of image)")
    print(f"📊 Background Coverage: {background_pixels:,} / {total_pixels:,} pixels ({background_percentage:.1f}% of image)")

if __name__ == "__main__":
    main() 