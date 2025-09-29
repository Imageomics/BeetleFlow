import torch
import os
import argparse
import glob
import cv2
import numpy as np
import evaluate
from tqdm import tqdm
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor
)

# Same as in config.py
LABEL_COLORS_LIST_RGB = [
    (0, 0, 0),          # background
    (200, 0, 0),        # head
    (0, 200, 0),        # pronotum
    (0, 0, 200),        # elytra
    (200, 200, 0),      # legs
    (200, 0, 200),      # antennas
]

def predict(model, processor, image, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    pred_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.shape[:2]]
    )[0]
    return pred_map

def draw_segmentation_map(labels, palette_rgb):
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(palette_rgb)):
        index = labels == label_num
        red_map[index] = np.array(palette_rgb)[label_num, 0]
        green_map[index] = np.array(palette_rgb)[label_num, 1]
        blue_map[index] = np.array(palette_rgb)[label_num, 2]

    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def image_overlay(image_bgr, segmented_image_bgr):
    alpha = 0.6
    beta = 1.0 - alpha
    gamma = 0

    image_bgr = np.array(image_bgr, dtype=np.uint8)

    overlaid_image = cv2.addWeighted(image_bgr, beta, segmented_image_bgr, alpha, gamma)
    return overlaid_image

def convert_rgb_mask_to_label(mask_rgb, palette_rgb):
    label_mask = np.zeros((mask_rgb.shape[0], mask_rgb.shape[1]), dtype=np.uint8)
    for i, color in enumerate(palette_rgb):
        locations = np.where(np.all(mask_rgb == color, axis=-1))
        label_mask[locations] = i
    return label_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='outputs/model_iou',
        help='Path to the trained model directory'
    )
    parser.add_argument(
        '--input',
        default='input',
        help='Path to the input directory containing test_images and test_masks'
    )
    parser.add_argument(
        '--output',
        default='output',
        help='Path to the output directory'
    )
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Compute device, e.g., "cuda:0" or "cpu"'
    )
    parser.add_argument(
        '--imgsz',
        default=[512, 512],
        type=int,
        nargs='+',
        help='[width, height] for image resizing'
    )
    args = parser.parse_args()
    print(f"--- Running evaluation with the following arguments ---\n{args}\n")

    # Create folders
    output_dir = 'testset_results'
    prediction_dir = os.path.join(output_dir, 'prediction')
    ground_truth_dir = os.path.join(output_dir, 'ground_truth')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(ground_truth_dir, exist_ok=True)
    print(f"Results will be saved in '{output_dir}' directory.")

    # Load model and metric
    device = torch.device(args.device)
    processor = Mask2FormerImageProcessor(do_reduce_labels=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model)
    model.to(device).eval()
    print("Model loaded successfully.")

    metric = evaluate.load("mean_iou")

    # Get all test images/masks
    test_images_dir = os.path.join(args.input, 'test_images')
    test_masks_dir = os.path.join(args.input, 'test_masks')
    image_paths = sorted(glob.glob(os.path.join(test_images_dir, '*')))

    if not image_paths:
        print(f"Error: No images found in '{test_images_dir}'. Please check the path.")
        return

    # Process each test image/mask
    for image_path in tqdm(image_paths, desc="Processing Test Set"):
        image_name = os.path.basename(image_path)

        # Get image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue
        image_bgr_resized = cv2.resize(image_bgr, (args.imgsz[0], args.imgsz[1]))
        image_rgb_resized = cv2.cvtColor(image_bgr_resized, cv2.COLOR_BGR2RGB)

        # Get mask
        mask_path = os.path.join(test_masks_dir, image_name)
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for image {image_name} at '{mask_path}'. Skipping.")
            continue
        gt_mask_bgr = cv2.imread(mask_path)
        gt_mask_bgr_resized = cv2.resize(gt_mask_bgr, (args.imgsz[0], args.imgsz[1]), interpolation=cv2.INTER_NEAREST)
        gt_mask_rgb_resized = cv2.cvtColor(gt_mask_bgr_resized, cv2.COLOR_BGR2RGB)

        # Predict
        predicted_labels = predict(model, processor, image_rgb_resized, device)
        predicted_labels = predicted_labels.cpu().numpy()

        # For metric calculation
        gt_labels = convert_rgb_mask_to_label(gt_mask_rgb_resized, LABEL_COLORS_LIST_RGB)
        metric.add_batch(predictions=[predicted_labels], references=[gt_labels])

        # Save image + prediction
        predicted_rgb_map = draw_segmentation_map(predicted_labels, LABEL_COLORS_LIST_RGB)
        predicted_bgr_map = cv2.cvtColor(predicted_rgb_map, cv2.COLOR_RGB2BGR)
        prediction_overlay = image_overlay(image_bgr_resized, predicted_bgr_map)
        cv2.imwrite(os.path.join(prediction_dir, image_name), prediction_overlay)

        # Save image + ground truth
        gt_overlay = image_overlay(image_bgr_resized, gt_mask_bgr_resized)
        cv2.imwrite(os.path.join(ground_truth_dir, image_name), gt_overlay)

    # Calculate metric
    num_classes = len(LABEL_COLORS_LIST_RGB)
    eval_metrics = metric.compute(
        num_labels=num_classes,
        ignore_index=255,
        reduce_labels=False
    )

    print("\n----------- Evaluation Results -----------")
    print(f"Mean IoU: {eval_metrics['mean_iou']:.4f}")
    print(f"Mean Accuracy: {eval_metrics['mean_accuracy']:.4f}")
    print("\nPer-category IoU:")
    for i, iou in enumerate(eval_metrics['per_category_iou']):
        # Same as in config.py
        class_names = ['background', 'head', 'pronotum', 'elytra', 'legs', 'antennas']
        print(f"  - {class_names[i]}: {iou:.4f}")
    print("------------------------------------------")
    print("\nEvaluation finished successfully.")

if __name__ == '__main__':
    main()