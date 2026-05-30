import numpy as np
import pickle
import os
from pathlib import Path


def read_kitti_calib(filepath):
    """Parses the KITTI calibration text file into numpy matrices."""
    calib = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            if line == '\n': continue
            key, value = line.split(':', 1)
            calib[key] = np.array([float(x) for x in value.strip().split()])

    calib['P2'] = calib['P2'].reshape(3, 4)
    calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)
    calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
    return calib


def project_lidar_to_camera(velo_path, calib, img_width, img_height):
    """Projects 3D LiDAR points to 2D Camera pixels."""
    velo_points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    points_3d = velo_points[:, :3]
    ones = np.ones((points_3d.shape[0], 1))
    points_3d_homo = np.hstack([points_3d, ones])

    Tr_velo_to_cam = np.vstack([calib['Tr_velo_to_cam'], [0, 0, 0, 1]])
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calib['R0_rect']
    P2 = calib['P2']

    proj_matrix = P2 @ R0_rect @ Tr_velo_to_cam
    points_2d_homo = proj_matrix @ points_3d_homo.T

    depths = points_2d_homo[2, :]
    valid_depth_mask = depths > 0

    u = points_2d_homo[0, :] / depths
    v = points_2d_homo[1, :] / depths

    valid_uv_mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    valid_mask = valid_depth_mask & valid_uv_mask

    projected_2d = np.vstack((u[valid_mask], v[valid_mask])).T

    # Return the physical 3D points (in Velodyne coordinates) that land on the camera
    valid_3d_points = points_3d[valid_mask]

    return projected_2d, valid_3d_points


# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":

    save_file = "saved_car_masks.pkl"
    if not os.path.exists(save_file):
        print(f"Error: Could not find {save_file}. Run Step 1 first!")
        exit()

    print(f"Loading cached masks from {save_file}...\n")
    with open(save_file, 'rb') as f:
        masks_dictionary = pickle.load(f)

    # Base directories
    base_dir_img = r"D:\Lidar and Radar\data_object_image_2\training\image_2"
    base_dir_calib = r"D:\Lidar and Radar\data_object_calib\training\calib"
    base_dir_lidar = r"D:\Lidar and Radar\data_object_velodyne\training\velodyne"

    output_folder = "output_extracted_cars"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Extraction data will be saved to: ./{output_folder}/")

    image_paths = sorted(Path(base_dir_img).glob("*.png"))

    for img_path in image_paths:
        frame_id = img_path.stem
        print(f"==================================================")
        print(f" PROCESSING FRAME: {frame_id}")
        print(f"==================================================")

        calib_file_path = os.path.join(base_dir_calib, f"{frame_id}.txt")
        lidar_file_path = os.path.join(base_dir_lidar, f"{frame_id}.bin")

        if not os.path.exists(calib_file_path) or not os.path.exists(lidar_file_path):
            continue

        car_masks = []
        for saved_path in masks_dictionary.keys():
            if saved_path.endswith(f"{frame_id}.png"):
                car_masks = masks_dictionary[saved_path]
                break

        if len(car_masks) == 0:
            print(f"  No cars found by YOLO in this frame. Skipping extraction.\n")
            continue

        img_h, img_w = car_masks[0].shape

        try:
            calib_data = read_kitti_calib(calib_file_path)
            points_2d, points_3d = project_lidar_to_camera(lidar_file_path, calib_data, img_w, img_h)
        except Exception as e:
            print(f"  Error projecting {frame_id}: {e}")
            continue

        u_coords = points_2d[:, 0].astype(int)
        v_coords = points_2d[:, 1].astype(int)

        for idx, mask in enumerate(car_masks):

            point_is_inside_mask = mask[v_coords, u_coords] > 0.5
            car_points_3d = points_3d[point_is_inside_mask]

            if len(car_points_3d) == 0:
                print(f"    -> Car {idx + 1}: Skipped (0 LiDAR points hit this YOLO mask).")
                continue

            # ---------------------------------------------------------
            # 3D SPATIAL OUTLIER REMOVAL (LiDAR Coordinates)
            # ---------------------------------------------------------
            med_x = np.median(car_points_3d[:, 0])  # Forward
            med_y = np.median(car_points_3d[:, 1])  # Side
            med_z = np.median(car_points_3d[:, 2])  # Height

            spatial_mask = (
                    (np.abs(car_points_3d[:, 0] - med_x) < 3.0) &
                    (np.abs(car_points_3d[:, 1] - med_y) < 1.5) &
                    (np.abs(car_points_3d[:, 2] - med_z) < 1.0)
            )
            filtered_points_3d = car_points_3d[spatial_mask]

            if len(filtered_points_3d) == 0:
                print(f"    -> Car {idx + 1}: Skipped (All points removed as background noise).")
                continue
            # ---------------------------------------------------------

            # ⬇️ NEW PERCENTILE MATH ⬇️
            # Calculate the array of all Euclidean distances for the clean points
            x_coords = filtered_points_3d[:, 0]
            y_coords = filtered_points_3d[:, 1]
            distances = np.sqrt(x_coords ** 2 + y_coords ** 2)

            # Extract percentiles
            dist_5th = np.percentile(distances, 5)
            dist_10th = np.percentile(distances, 10)
            dist_50th = np.percentile(distances, 50)

            # Set the official measurement to the robust 5th percentile
            official_distance = dist_5th

            print(f"    -> Car {idx + 1}: Found {len(filtered_points_3d):>4} clean points.")
            print(f"         5th % (Bumper): {dist_5th:.2f}m | 10th %: {dist_10th:.2f}m | 50th %: {dist_50th:.2f}m")

            # Save the clean 3D points to a text file using the 5th percentile distance
            save_filename = f"{frame_id}_car_{idx + 1}_dist_{official_distance:.2f}m.txt"
            save_path = os.path.join(output_folder, save_filename)

            np.savetxt(save_path, filtered_points_3d, fmt="%.4f", header="X Y Z", comments="")

        print("\n")