# DeepWeeds Dataset

## Overview
DeepWeeds is a comprehensive dataset consisting of 17,509 images capturing eight different weed species native to Australia in situ with neighbouring flora. The dataset was originally published in Scientific Reports and has achieved an average accuracy of 95.7% using the ResNet50 deep convolutional neural network.

## Dataset Details
- **Total Images**: 17,509
- **Image Format**: RGB images (256x256x3)
- **Classes**: 9 (8 weed species + 1 negative class)
- **Download Size**: 469.32 MiB
- **Dataset Size**: 469.99 MiB

## Weed Species
The dataset includes the following weed species native to Queensland, Australia:
1. Chinee apple
2. Snake weed
3. Lantana
4. Prickly acacia
5. Siam weed
6. Parthenium
7. Rubber vine
8. Parkinsonia
9. Negative (non-weed class)

## Collection Locations
Images were collected from weed infestations across Queensland at the following sites:
- Black River
- Charters Towers
- Cluden
- Douglas
- Hervey Range
- Kelso
- McKinlay
- Paluma

## Data Organization
- Images are named using the format: `YYYYMMDD-HHMMSS-ID`
  - Example: `20170320-093423-1`
- Labels are provided in CSV format with columns: Filename, Label, Species

## Accessing the Dataset
The dataset can be accessed in multiple ways:

1. **Direct Download**:
   - [images.zip](https://drive.google.com/file/d/1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj) (468 MB)
   - [models.zip](https://drive.google.com/file/d/1MRbN5hXOTYnw7-71K-2vjY01uJ9GkQM5) (477 MB)

2. **TensorFlow Datasets**:
   The dataset is available through TensorFlow Datasets (TFDS) catalog.

## License
- Images and annotations: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Source code: [Apache 2](LICENSE)

## Citation
If you use the DeepWeeds dataset in your work, please cite:

```bibtex
@article{DeepWeeds2019,
  author = {Alex Olsen and
    Dmitry A. Konovalov and
    Bronson Philippa and
    Peter Ridd and
    Jake C. Wood and
    Jamie Johns and
    Wesley Banks and
    Benjamin Girgenti and
    Owen Kenny and 
    James Whinney and
    Brendan Calvert and
    Mostafa {Rahimi Azghadi} and
    Ronald D. White},
  title = {{DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning}},
  journal = {Scientific Reports},
  year = 2019,
  number = 2058,
  month = 2,
  volume = 9,
  issue = 1,
  day = 14,
  url = "https://doi.org/10.1038/s41598-018-38343-3",
  doi = "10.1038/s41598-018-38343-3"
}
```

## Original Paper
For more details, please refer to the original paper:
[DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning](https://www.nature.com/articles/s41598-018-38343-3)

## Repository
The original repository can be found at: [https://github.com/AlexOlsen/DeepWeeds/](https://github.com/AlexOlsen/DeepWeeds/) 