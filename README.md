# BeetleFlow: An Integrative Deep Learning Pipeline for Beetle Image Processing

![main figure](assets/pipeline.png)

## 🗓️ TODO
- [ ] Provide checkpoint downloads

## 📊 Dataset
Beetle part segmentation dataset is available [here](data/) (both 5-class and 9-class).

## 🧑‍💻 Usage

### Detection

```bash
python pipeline/1_detection/detection_beetles.py \
    --image-dir=/path/to/input/tray/image/dir \
    --output-dir=/path/to/output/dir \
    --metadata-file /path/to/all/individual/metadata.csv
```

### Cropping

Modify the paths in `1_sort_and_crop.py`, `2_match_metadata.py`, and `3_check_matching.py`, then run:
```bash
python 1_sort_and_crop.py
python 2_match_metadata.py
python 3_check_matching.py
```

### Segmentation

Download the checkpoints, then run:
```bash
python batch_inference.py \
    --input_dir /path/to/cropped/images/dir \
    --output_dir /path/to/output/dir \
    --model ./checkpoints/5-class # or 9-class
```