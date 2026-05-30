import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import pickle
import os


def process_kitti_image_folder(folder_path, model_path='yolo26m-seg.pt'):
    """
    Processes a folder of images using YOLO segmentation and returns a dictionary
    mapping file paths to their respective car masks.
    """
    model = YOLO(model_path)
    image_dir = Path(folder_path)
    all_car_masks = {}

    print(f"Starting YOLO processing on folder: {folder_path}")
    print("This might take a while depending on the number of images...")

    # Iterate through all PNG images
    for img_path in sorted(image_dir.glob('*.png')):
        img_path_str = str(img_path)

        image = cv2.imread(img_path_str)
        if image is None:
            continue

        img_height, img_width = image.shape[:2]

        # Run inference (verbose=False keeps the console clean)
        results = model(image, verbose=False)
        car_masks = []

        for result in results:
            if result.masks is not None:
                for mask_tensor, box in zip(result.masks.data, result.boxes):
                    class_id = int(box.cls[0])

                    if class_id == 2:  # Class 2 is 'car' in COCO dataset
                        # Convert tensor to numpy array and resize to original image dimensions
                        mask = mask_tensor.cpu().numpy()
                        mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                        car_masks.append(mask)

        all_car_masks[img_path_str] = car_masks

        # Print progress every 100 images so you know it's working
        if len(all_car_masks) % 100 == 0:
            print(f"Processed {len(all_car_masks)} images...")

    print(f"Finished processing. Total images read: {len(all_car_masks)}")
    return all_car_masks


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":

    # Using raw string (r) to handle Windows paths
    dataset_folder = r"D:\Lidar and Radar\data_object_image_2\training\image_2/"
    save_file = "saved_car_masks.pkl"

    # ⬇️ Define and create the output folder for drawn images ⬇️
    output_folder = "output_bounding_boxes"
    os.makedirs(output_folder, exist_ok=True)

    # ---------------------------------------------------------
    # 1. SAVE / LOAD LOGIC
    # ---------------------------------------------------------
    if os.path.exists(save_file):
        print(f"Found existing saved results at '{save_file}'.")
        print("Loading directly into memory...")
        with open(save_file, 'rb') as f:
            masks_dictionary = pickle.load(f)
        print("Results loaded successfully!")
    else:
        print("No saved results found. Running YOLO segmentation...")
        masks_dictionary = process_kitti_image_folder(dataset_folder)
        with open(save_file, 'wb') as f:
            pickle.dump(masks_dictionary, f)
        print("Results successfully saved to your hard drive!")

    # ---------------------------------------------------------
    # 2. ALL 20 IMAGES & BOUNDING BOX EXTRACTION + DRAWING
    # ---------------------------------------------------------
    print("\n==================================================")
    print(" BATCH PROCESSING, BOUNDING BOX EXTRACTION & DRAWING")
    print("==================================================")
    print(f"Images will be saved to: ./{output_folder}/")

    # Loop through every single image in the dictionary
    for img_path_str, car_masks in masks_dictionary.items():

        # Extract just the filename (e.g., "006037.png") from the long path
        frame_filename = Path(img_path_str).name

        print(f"\n--- Frame: {frame_filename} | Total Cars Found: {len(car_masks)} ---")

        # Load the original image so we can draw on it
        image_to_draw = cv2.imread(img_path_str)
        if image_to_draw is None:
            print(f"  Warning: Could not read original image to draw on. Skipping visual save.")
            continue

        # Loop through every car found in this specific frame
        for idx, mask in enumerate(car_masks):

            # Find the (Y, X) coordinates of every pixel that belongs to the car
            y_coords, x_coords = np.where(mask > 0.5)

            # Safety check: Ensure the mask isn't completely empty
            if len(y_coords) > 0:
                # Calculate Bounding Box bounds
                x_min = np.min(x_coords)
                x_max = np.max(x_coords)
                y_min = np.min(y_coords)
                y_max = np.max(y_coords)

                total_pixels = len(y_coords)

                # Print the data cleanly
                print(f"  -> Car {idx + 1}:")
                print(f"       Mask Matrix Shape: {mask.shape} (Total Physical Car Pixels: {total_pixels})")
                print(f"       Bounding Box:      [X_min: {x_min}, Y_min: {y_min}, X_max: {x_max}, Y_max: {y_max}]")

                # Draw the bounding box (Green, thickness 2)
                cv2.rectangle(image_to_draw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Add a label right above the box
                label_text = f"Car {idx + 1}"
                cv2.putText(image_to_draw, label_text, (x_min, y_min - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                            2)

            else:
                print(f"  -> Car {idx + 1}: Empty mask matrix! (False positive)")

        # Save the fully drawn image to the output folder
        output_image_path = os.path.join(output_folder, frame_filename)
        cv2.imwrite(output_image_path, image_to_draw)
        print(f"  => Saved visualized image to {output_image_path}")

    print("\nProcessing complete! Check the 'output_bounding_boxes' folder to see your results.")