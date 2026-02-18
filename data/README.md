# Beetle Part Segmentation Dataset

These images are sourced from the [Beetles as Sentinel Taxa: Predicting drought conditions from NEON specimen imagery dataset](https://huggingface.co/datasets/imageomics/sentinel-beetles) and are licensed under [Creative Commons Attribution (CC BY) 4.0](https://creativecommons.org/licenses/by/4.0/). If you use them in your research, please be sure to cite them appropriately (see [Acknowledgments](../README.md#acknowledgments)).

## Dataset Statistics

### 5-class

- Segmented parts: head, pronotum, elytra, legs, and antennas

- 340 labeled beetles with RGB masks

- Train/test split:
    - Training set: 272 images
    - Test set: 68 images

- Mask RGB values:
    - Head: (200, 0, 0)
    - Pronotum: (0, 200, 0)
    - Elytra: (0, 0, 200)
    - Legs: (200, 200, 0)
    - Antennas: (200, 0, 200)

### 9-class

- Segmented parts: head, pronotum, elytra, legs, antennas, eyes, mouthparts, tail, and pin

- 330 labeled beetles with RGB masks

- Train/test split:
    - Training set: 264 images
    - Test set: 66 images

- Mask RGB values:
    - Head: (100, 0, 0)
    - Eyes: (0, 100, 0)
    - Mouthparts: (0, 0, 100)
    - Pronotum: (100, 100, 0)
    - Elytra: (100, 0, 100)
    - Tail: (0, 100, 100)
    - Legs: (200, 0, 0)
    - Antennas: (0, 200, 0)
    - Pin: (0, 0, 200)

## Scripts

- `mask_visualization.py`: Visualize an image overlaid by its mask.
- `mask_conversion.py`: If you manually label more beetle images where the output labels are in `json` format (like when using **AnyLabeling** and **X-AnyLabeling**), you can use this script to convert `json` files into RGB `png` images. In the script, you need to modify `label_to_pixel_value` to your custom label tags.
