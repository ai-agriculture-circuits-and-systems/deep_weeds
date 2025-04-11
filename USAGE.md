# DeepWeeds Dataset Usage Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Applications](#applications)
3. [Model Training](#model-training)
4. [Inference](#inference)
5. [Performance Metrics](#performance-metrics)

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow
- Required packages (install using `pip install -r requirements.txt`)

### Dataset Setup
1. Download the dataset:
   ```bash
   # Using direct download
   wget https://drive.google.com/file/d/1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj -O images.zip
   unzip images.zip

   # Using TensorFlow Datasets
   import tensorflow_datasets as tfds
   dataset = tfds.load('deep_weeds', split='train')
   ```

## Applications

### 1. Weed Species Classification
The primary application of the DeepWeeds dataset is for training models to classify different weed species. This can be used for:
- Automated weed detection in agricultural fields
- Precision agriculture systems
- Environmental monitoring
- Research on weed species distribution

### 2. Agricultural Robotics
The dataset can be used to train models for:
- Autonomous weed detection systems
- Robotic weed control systems
- Precision spraying systems
- Crop monitoring drones

### 3. Environmental Research
Applications include:
- Studying weed distribution patterns
- Monitoring invasive species
- Environmental impact assessment
- Biodiversity studies

### 4. Educational Purposes
The dataset is suitable for:
- Teaching computer vision concepts
- Training students in machine learning
- Demonstrating multi-class classification
- Understanding agricultural AI applications

## Model Training

### Using the Provided Script
The repository includes `deepweeds.py` for training and evaluation:

```bash
# Train ResNet50 with five-fold cross validation
python3 deepweeds.py cross_validate --model resnet

# Train InceptionV3 with five-fold cross validation
python3 deepweeds.py cross_validate --model inception
```

### Custom Training
You can also train your own models:

```python
import tensorflow as tf
import tensorflow_datasets as tfds

# Load dataset
dataset = tfds.load('deep_weeds', split='train')

# Preprocess data
def preprocess(features):
    image = tf.cast(features['image'], tf.float32) / 255.0
    label = features['label']
    return image, label

dataset = dataset.map(preprocess).batch(32)

# Create and train model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(9, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(dataset, epochs=10)
```

## Inference

### Using Pre-trained Models
The repository provides pre-trained models for inference:

```bash
# Measure inference times for ResNet50
python3 deepweeds.py inference --model models/resnet.hdf5

# Measure inference times for InceptionV3
python3 deepweeds.py inference --model models/inception.hdf5
```

### TensorRT Integration
For NVIDIA Jetson TX2 platforms:
```bash
cd tensorrt/src
make -j4
cd ../bin
./resnet_inference
```

## Performance Metrics

### Model Performance
- ResNet50 achieved 95.7% average accuracy
- InceptionV3 also provides competitive performance
- Models are evaluated using five-fold cross-validation

### Inference Speed
- ResNet50 TensorRT implementation available for Jetson TX2
- Benchmark results available in the original paper
- Custom implementations can be optimized for specific hardware

## Best Practices

1. **Data Preprocessing**
   - Normalize images to [0,1] range
   - Apply data augmentation for better generalization
   - Consider class imbalance in the dataset

2. **Model Selection**
   - ResNet50 recommended for best accuracy
   - Consider model size vs. inference speed trade-offs
   - Use transfer learning for faster training

3. **Evaluation**
   - Use five-fold cross-validation
   - Consider both accuracy and inference time
   - Test on real-world scenarios

4. **Deployment**
   - Optimize models for target hardware
   - Consider using TensorRT for NVIDIA platforms
   - Implement proper error handling

## Additional Resources

- [Original Paper](https://www.nature.com/articles/s41598-018-38343-3)
- [TensorFlow Datasets Documentation](https://www.tensorflow.org/datasets/catalog/deep_weeds)
- [GitHub Repository](https://github.com/AlexOlsen/DeepWeeds/) 