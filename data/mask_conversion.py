import os
import cv2
import json
import numpy as np

def convert_mask_json2png(json_file_path, output_mask_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_width = data['imageWidth']
    img_height = data['imageHeight']
    shapes = data['shapes']

    mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    label_to_pixel_value = {
        # "label": (B, G, R)
        "head": (0, 0, 200),            # RGB: (200, 0, 0)
        "pronotum": (0, 200, 0),        # RGB: (0, 200, 0)
        "elytra": (200, 0, 0),          # RGB: (0, 0, 200)
        "legs": (0, 200, 200),          # RGB: (200, 200, 0)
        "antennas": (200, 0, 200)       # RGB: (200, 0, 200)
    }

    # label_to_pixel_value = {
    #     # "label": (B, G, R)
    #     "head": (0, 0, 100),            # RGB: (100, 0, 0)
    #     "eyes": (0, 100, 0),            # RGB: (0, 100, 0)
    #     "mouth_parts": (100, 0, 0),     # RGB: (0, 0, 100)
    #     "pronotum": (0, 100, 100),      # RGB: (100, 100, 0)
    #     "elytra": (100, 0, 100),        # RGB: (100, 0, 100)
    #     "tail": (100, 100, 0),          # RGB: (0, 100, 100)
    #     "legs": (0, 0, 200),            # RGB: (200, 0, 0)
    #     "antennas": (0, 200, 0),        # RGB: (0, 200, 0)
    #     "pin": (200, 0, 0),             # RGB: (0, 0, 200)
    # }

    assigned_labels = set()

    for shape in shapes:
        label = shape['label']
        polygon_points = shape['points']
        shape_type = shape['shape_type']

        if shape_type != "polygon":
            print(f"Warning: skipping non-polygon shape '{label}' (type: {shape_type}).")
            continue

        if label not in label_to_pixel_value:
            if label not in assigned_labels:
                print(f"Warning: label '{label}' is not defined, skipping. Please define in 'label_to_pixel_value'.")
                assigned_labels.add(label)
            continue

        pixel_value = label_to_pixel_value[label]
        pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], pixel_value)

    cv2.imwrite(output_mask_path, mask)
    print(f"Finish converting and saving: {output_mask_path}")

if __name__ == "__main__":
    json_input_dir = 'input/train_jsons'
    png_output_dir = 'input/train_masks'

    for json_name in os.listdir(json_input_dir):
        json_file_path = os.path.join(json_input_dir, json_name)
        output_mask_path = os.path.join(png_output_dir, f"{os.path.splitext(json_name)[0]}.png")
        convert_mask_json2png(json_file_path, output_mask_path)