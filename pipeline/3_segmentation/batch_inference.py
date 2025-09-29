import torch
import os
import argparse
import glob
import cv2
import numpy as np
from tqdm import tqdm
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor
)

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

    # Resize prediction result to original image size
    pred_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.shape[:2]]
    )[0]
    return pred_map

def draw_segmentation_map(labels, palette_rgb):
    color_map = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    for label_num, color in enumerate(palette_rgb):
        locations = labels == label_num
        color_map[locations] = color

    return color_map

def image_overlay(image_bgr, segmented_image_bgr):
    alpha = 0.8     # Transparency of mask
    beta = 1.0      # Transparency of image
    gamma = 0

    if image_bgr.shape != segmented_image_bgr.shape:
        segmented_image_bgr = cv2.resize(segmented_image_bgr, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlaid_image = cv2.addWeighted(image_bgr, beta, segmented_image_bgr, alpha, gamma)
    return overlaid_image


def main():
    parser = argparse.ArgumentParser(description="Batch inference script for beetle segmentation.")
    parser.add_argument(
        '--model',
        default='outputs/model_iou',
        help='Path to the trained model directory.'
    )
    parser.add_argument(
        '--input_dir',
        default='images',
        help='Main input directory containing subfolders with beetle images.'
    )
    parser.add_argument(
        '--output_dir',
        default='image_predictions',
        help='Main output directory to save results.'
    )
    parser.add_argument(
        '--device',
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
        help='Compute device, e.g., "cuda:0" or "cpu".'
    )
    args = parser.parse_args()
    print(f"--- Running Batch Inference ---\n{args}\n")

    # Load model
    device = torch.device(args.device)
    processor = Mask2FormerImageProcessor(do_reduce_labels=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(args.model)
    model.to(device).eval()
    print(f"Model loaded successfully and running on {device}.")

    # Look for all folders to process
    search_path = os.path.join(args.input_dir, '**', 'single_beetles')
    beetle_folders = glob.glob(search_path, recursive=True)

    if not beetle_folders:
        print(f"Error: No 'single_beetles' folders found inside '{args.input_dir}'.")
        return

    print(f"Found {len(beetle_folders)} 'single_beetles' folders to process.")

    predicted_masks_dir = os.path.join(args.output_dir, 'predicted_masks')
    predicted_images_dir = os.path.join(args.output_dir, 'predicted_mask_images')
    os.makedirs(predicted_masks_dir, exist_ok=True)
    os.makedirs(predicted_images_dir, exist_ok=True)

    # Process each folder
    num_inference = 0
    for single_beetle_folder in tqdm(beetle_folders, desc="Processing Folders"):
        # single_beetle_folder: 'images/folder1/single_beetles'
        # subfolder_name: folder1
        subfolder_name = os.path.basename(os.path.dirname(single_beetle_folder))
        image_paths = glob.glob(os.path.join(single_beetle_folder, '*'))
        for image_path in tqdm(image_paths, desc=f"Images in {subfolder_name}", leave=False):
            try:
                image_name = os.path.basename(image_path)
                image_ext = os.path.splitext(image_name)[1]
                image_bgr = cv2.imread(image_path)
                if image_bgr is None:
                    print(f"Warning: Could not read {image_path}. Skipping.")
                    continue
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                # Inference
                predicted_labels = predict(model, processor, image_rgb, device)
                predicted_labels_np = predicted_labels.cpu().numpy()

                # Save mask
                predicted_rgb_mask = draw_segmentation_map(predicted_labels_np, LABEL_COLORS_LIST_RGB)
                predicted_bgr_mask = cv2.cvtColor(predicted_rgb_mask, cv2.COLOR_RGB2BGR)
                mask_save_path = os.path.join(predicted_masks_dir, f"{num_inference}{image_ext}")
                cv2.imwrite(mask_save_path, predicted_bgr_mask)

                # Save mask + image
                overlay_image = image_overlay(image_bgr, predicted_bgr_mask)
                overlay_save_path = os.path.join(predicted_images_dir, f"{num_inference}{image_ext}")
                cv2.imwrite(overlay_save_path, overlay_image)

            except Exception as e:
                print(f"\nAn error occurred while processing {image_path}: {e}")

        num_inference += 1

    print("\n--- Batch inference finished. ---")

if __name__ == '__main__':
    main()