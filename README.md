# EOMT Panoptic Segmentation Image Splitter

A Python tool that uses the EOMT (End-to-End Multi-Task) model to perform panoptic segmentation on images and extract individual segmented regions as separate files.

## 🎯 What It Does

This tool takes an input image and:

1. **Performs Panoptic Segmentation**: Uses the EOMT model to identify and segment different objects/regions in the image
2. **Extracts Individual Segments**: Creates separate image files for each detected segment
3. **Preserves Original Dimensions**: Each extracted segment maintains the same dimensions as the original image
4. **Uses Accurate Labels**: Leverages the model's built-in COCO panoptic label mapping for precise object identification
5. **Optimizes Performance**: Automatically uses the best available device (CUDA, MPS, or CPU)

## 🚀 Features

- **Automatic Device Detection**: Uses CUDA (NVIDIA), MPS (Apple Silicon), or CPU
- **Accurate Label Mapping**: Uses model's official COCO panoptic labels
- **Clean File Output**: Saves individual segments with descriptive names
- **Detailed Statistics**: Provides pixel counts and percentages for each segment
- **Black Background**: Non-segment areas are filled with black for clean extraction

## 📋 Requirements

### Python Dependencies
```bash
pip install torch torchvision
pip install transformers
pip install pillow
pip install numpy
pip install scipy
```

### Hardware Requirements
- **CUDA**: NVIDIA GPU with CUDA support (recommended)
- **MPS**: Apple Silicon Mac (M1/M2/M3) with macOS 12.3+
- **CPU**: Any system (fallback option)

## 🛠️ Installation

1. **Clone or download** the script files
2. **Install dependencies**:
   ```bash
   pip install torch torchvision transformers pillow numpy
   ```
3. **Ensure you have the required image file** in the specified path

## 📖 Usage

### Basic Usage

1. **Update the image path** in `segment_splitter.py`:
   ```python
   image_path = "/path/to/your/image.jpg"
   ```

2. **Run the script**:
   ```bash
   python coco_panoptic_eomt_large_640/segment_splitter.py
   ```

### Example Output

```
Using device: mps
Original image size: (800, 541)
Loading EOMT model...
Model moved to mps

Found 4 segments:
  ID 1: person (label_id: 1)
  ID 2: cat (label_id: 16)
  ID 3: chair (label_id: 57)
  ID 4: table (label_id: 61)

Extracting segments to 'segmented_images' directory...
Saving all 4 segments...
  Saved: person_id_1.png
  Saved: cat_id_2.png
  Saved: chair_id_3.png
  Saved: table_id_4.png

✅ All segments saved to 'segmented_images' directory!
📊 Summary:
   - Original image: 800x541 pixels
   - Segments found: 4
   - Files created: 4

📋 Segment Details:
   - person (ID: 1, label_id: 1): 29,347 pixels (6.8%)
   - cat (ID: 2, label_id: 16): 50,500 pixels (11.7%)
   - chair (ID: 3, label_id: 57): 158,307 pixels (36.6%)
   - table (ID: 4, label_id: 61): 154,463 pixels (35.7%)
```

## 📁 Output Structure

The script creates a `segmented_images/` directory containing:

```
segmented_images/
├── person_id_1.png          # Person segment
├── cat_id_2.png            # Cat segment  
├── chair_id_3.png          # Chair segment
├── table_id_4.png          # Table segment
└── ... (other segments)
```

### File Naming Convention
- **Format**: `{label_name}_id_{segment_id}.png`
- **Example**: `cat_id_2.png` where:
  - `cat` = detected object label
  - `2` = unique segment ID

## 🔧 Technical Details

### Model Information
- **Model**: `tue-mps/coco_panoptic_eomt_large_640`
- **Type**: EOMT (End-to-End Multi-Task) for Universal Segmentation
- **Dataset**: COCO Panoptic Segmentation
- **Labels**: 133 COCO panoptic classes (80 things + 53 stuff)

### Device Optimization
The script automatically detects and uses the best available device:

1. **CUDA** (NVIDIA GPUs): 5-10x faster than CPU
2. **MPS** (Apple Silicon): 3-5x faster than CPU
3. **CPU**: Reliable fallback for all systems

### Segmentation Process
1. **Image Processing**: Converts image to model-compatible format
2. **Model Inference**: Runs EOMT model on the image
3. **Post-Processing**: Converts model output to segmentation masks
4. **Segment Extraction**: Creates individual files for each detected region
5. **Label Mapping**: Uses model's built-in COCO label vocabulary

## 🎨 Supported Image Formats

- **JPEG** (.jpg, .jpeg)
- **PNG** (.png)
- **WebP** (.webp)
- **Other formats** supported by PIL/Pillow

## 📊 Supported Object Classes

The model can detect 133 different object classes including:

### Things (80 classes)
- **Animals**: person, cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Vehicles**: car, truck, bus, motorcycle, bicycle, airplane, boat, train
- **Objects**: chair, table, bed, tv, laptop, phone, book, clock, vase
- **Food**: apple, banana, orange, pizza, cake, sandwich, hot dog
- **And many more...**

### Stuff (53 classes)
- **Surfaces**: floor, ceiling, wall, pavement, grass, dirt, mud
- **Nature**: sky, clouds, mountain, hill, tree, flower, bush
- **Materials**: metal, plastic, wood, cloth, paper, glass
- **And many more...**

## 🔍 Troubleshooting

### Common Issues

1. **Device Mismatch Error**:
   ```
   RuntimeError: input(device='cpu') and weight(device='mps:0') must be on the same device
   ```
   - **Solution**: The script automatically handles this, but ensure you're using the latest version

2. **Memory Issues**:
   - **Solution**: Use smaller images or switch to CPU if GPU memory is insufficient

3. **Model Download Issues**:
   - **Solution**: Ensure stable internet connection for first-time model download

### Performance Tips

- **Use GPU**: CUDA or MPS for significantly faster processing
- **Image Size**: Larger images take longer to process
- **Batch Processing**: Run multiple images sequentially for efficiency

## 🤝 Contributing

Feel free to:
- Report bugs or issues
- Suggest improvements
- Add support for additional features
- Optimize performance

## 📄 License

This project uses the EOMT model which is subject to its own license terms. Please refer to the model's documentation for licensing information.

## 🙏 Acknowledgments

- **EOMT Model**: Developed by TU Eindhoven
- **COCO Dataset**: Microsoft COCO dataset for training data
- **Transformers Library**: Hugging Face for model implementation
- **PyTorch**: Facebook for the deep learning framework 