import torch
import torch.nn as nn
from transformers import EomtForUniversalSegmentation, AutoImageProcessor
import coremltools as ct
import numpy as np
from PIL import Image

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class EOMTWrapper(nn.Module):
    """Wrapper class for EOMT model to make it CoreML compatible"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, pixel_values):
        """Forward pass that returns the appropriate output for CoreML conversion"""
        outputs = self.model(pixel_values=pixel_values)
        
        # Check what attributes are available in the output
        if hasattr(outputs, 'logits'):
            return outputs.logits
        elif hasattr(outputs, 'segmentation_logits'):
            return outputs.segmentation_logits
        elif hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        else:
            # If none of the expected attributes exist, return the raw output
            # This might be a tuple or other structure
            if isinstance(outputs, tuple):
                return outputs[0]  # Return first element if it's a tuple
            else:
                return outputs

def create_sample_input(processor, device, image_size=(640, 640)):
    """Create a sample input for CoreML conversion"""
    # Create a dummy image
    dummy_image = Image.new('RGB', image_size, color='white')
    
    # Process the image
    inputs = processor(
        images=dummy_image,
        return_tensors="pt",
    )
    
    # Move to the same device as the model
    pixel_values = inputs['pixel_values'].to(device)
    return pixel_values

def debug_model_output(model, sample_input):
    """Debug function to inspect model output structure"""
    print("🔍 Debugging model output structure...")
    
    with torch.inference_mode():
        outputs = model(pixel_values=sample_input)
    
    print(f"Output type: {type(outputs)}")
    print(f"Output attributes: {dir(outputs)}")
    
    if hasattr(outputs, '__dict__'):
        print(f"Output dict keys: {outputs.__dict__.keys()}")
    
    # Try to access common attributes
    for attr in ['logits', 'segmentation_logits', 'last_hidden_state', 'hidden_states', 'attentions']:
        if hasattr(outputs, attr):
            value = getattr(outputs, attr)
            print(f"  {attr}: {type(value)}, shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
    
    return outputs

def export_to_coreml():
    """Export the EOMT model to CoreML format"""
    print("🚀 Starting EOMT to CoreML export...")
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load the model
    print("📥 Loading EOMT model...")
    model_id = "tue-mps/coco_panoptic_eomt_large_640"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = EomtForUniversalSegmentation.from_pretrained(model_id)
    
    # Move model to device
    model = model.to(device)
    print(f"Model moved to {device}")
    
    # Create wrapper
    wrapped_model = EOMTWrapper(model)
    wrapped_model.eval()
    
    # For CoreML conversion, we might need to use CPU
    print("🔄 Moving model to CPU for CoreML conversion...")
    wrapped_model = wrapped_model.cpu()
    
    # Create sample input on CPU
    print("🔧 Creating sample input...")
    sample_input = create_sample_input(processor, torch.device("cpu"))
    print(f"Sample input shape: {sample_input.shape}")
    
    # Debug model output structure
    debug_model_output(model, sample_input)
    
    # Trace the model
    print("📝 Tracing model...")
    traced_model = torch.jit.trace(wrapped_model, sample_input)
    
    # Convert to CoreML
    print("🔄 Converting to CoreML...")
    try:
        # Convert using coremltools
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="pixel_values", shape=sample_input.shape)],
            minimum_deployment_target=ct.target.iOS15,  # Adjust based on your needs
            compute_units=ct.ComputeUnit.CPU_AND_NE  # Use CPU and Neural Engine
        )
        
        # Save the model
        output_path = "eomt_panoptic_segmentation.mlpackage"
        coreml_model.save(output_path)
        print(f"✅ CoreML model saved to: {output_path}")
        
        # Print model information
        print("\n📊 CoreML Model Information:")
        print(f"   - Model file: {output_path}")
        print(f"   - Input shape: {sample_input.shape}")
        print(f"   - Deployment target: iOS 15+")
        print(f"   - Compute units: CPU + Neural Engine")
        print(f"   - Model type: ML Program (.mlpackage)")
        
        # Test the model
        print("\n🧪 Testing CoreML model...")
        test_input = sample_input.numpy()
        prediction = coreml_model.predict({"pixel_values": test_input})
        print(f"   - Output keys: {list(prediction.keys())}")
        if 'logits' in prediction:
            print(f"   - Output shape: {prediction['logits'].shape}")
        
        print("\n🎉 CoreML export completed successfully!")
        print("\n📱 Usage in iOS/macOS:")
        print("   1. Add the .mlpackage file to your Xcode project")
        print("   2. Use Vision framework or CoreML directly")
        print("   3. Preprocess images to match input requirements")
        print("   4. Post-process outputs for segmentation results")
        
    except Exception as e:
        print(f"❌ Error during CoreML conversion: {e}")
        print("\n🔄 Trying alternative conversion approach...")
        
        try:
            # Alternative: Try with different settings
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="pixel_values", shape=sample_input.shape)],
                minimum_deployment_target=ct.target.iOS16,
                compute_units=ct.ComputeUnit.CPU_ONLY
            )
            
            # Save the model
            output_path = "eomt_panoptic_segmentation.mlpackage"
            coreml_model.save(output_path)
            print(f"✅ CoreML model saved to: {output_path}")
            
        except Exception as e2:
            print(f"❌ Alternative conversion also failed: {e2}")
            print("\n🔧 Troubleshooting tips:")
            print("   - Ensure coremltools is installed: pip install coremltools")
            print("   - Check if model architecture is CoreML compatible")
            print("   - Try different deployment targets")
            print("   - Consider using a smaller model variant")
            print("   - The EOMT model might not be fully compatible with CoreML")
            print("   - Consider using ONNX export as an alternative")
            return

def create_ios_integration_guide():
    """Create a guide for iOS integration"""
    guide_content = """
# iOS Integration Guide for EOMT CoreML Model

## 1. Add Model to Xcode Project
- Drag `eomt_panoptic_segmentation.mlpackage` into your Xcode project
- Ensure "Add to target" is checked for your app target

## 2. Import Required Frameworks
```swift
import CoreML
import Vision
import UIKit
```

## 3. Basic Usage Example
```swift
class SegmentationManager {
    private var model: VNCoreMLModel?
    
    init() {
        do {
            model = try VNCoreMLModel(for: EOMTPanopticSegmentation().model)
        } catch {
            print("Failed to load model: \(error)")
        }
    }
    
    func segmentImage(_ image: UIImage, completion: @escaping (UIImage?) -> Void) {
        guard let model = model else {
            completion(nil)
            return
        }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNCoreMLFeatureValueObservation],
                  let firstResult = results.first,
                  let multiArray = firstResult.featureValue.multiArrayValue else {
                completion(nil)
                return
            }
            
            // Process the segmentation results
            let segmentedImage = self.processSegmentationResult(multiArray, originalImage: image)
            completion(segmentedImage)
        }
        
        request.imageCropAndScaleOption = .centerCrop
        
        let handler = VNImageRequestHandler(cgImage: image.cgImage!, options: [:])
        try? handler.perform([request])
    }
    
    private func processSegmentationResult(_ multiArray: MLMultiArray, originalImage: UIImage) -> UIImage? {
        // Implement segmentation post-processing here
        // Convert MLMultiArray to segmentation mask
        // Overlay on original image
        return nil // Placeholder
    }
}
```

## 4. Input Requirements
- **Image size**: 640x640 pixels (model default)
- **Color space**: RGB
- **Data type**: Float32
- **Normalization**: Apply the same preprocessing as the original model

## 5. Output Processing
- **Output shape**: [1, num_classes, height, width]
- **Post-processing**: Apply argmax to get segmentation IDs
- **Label mapping**: Use COCO panoptic labels for object identification

## 6. Performance Optimization
- Use Neural Engine when available
- Consider batch processing for multiple images
- Implement proper memory management
- Use background queues for processing

## 7. Error Handling
```swift
enum SegmentationError: Error {
    case modelNotLoaded
    case invalidInput
    case processingFailed
    case outputProcessingFailed
}
```

## 8. Memory Management
- Release model resources when not in use
- Monitor memory usage during processing
- Implement proper cleanup in deinit
"""
    
    with open("ios_integration_guide.md", "w") as f:
        f.write(guide_content)
    
    print("📖 iOS integration guide created: ios_integration_guide.md")

if __name__ == "__main__":
    try:
        export_to_coreml()
        create_ios_integration_guide()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Install required packages:")
        print("   pip install coremltools")
        print("   pip install torch torchvision")
        print("   pip install transformers")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("🔧 Check your environment and dependencies") 