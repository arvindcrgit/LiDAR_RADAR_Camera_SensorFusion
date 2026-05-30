import numpy as np
import pickle
import os
import cv2
from pathlib import Path


def read_kitti_calib(filepath):
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
    valid_3d_points = points_3d[valid_mask]

    return projected_2d, valid_3d_points


if __name__ == "__main__":
    save_file = "saved_car_masks.pkl"

    if not os.path.exists(save_file):
        print(f"Error: Could not find {save_file}. Run step1_segmentation.py first!")
        exit()

    print(f"Loading cached masks from {save_file}...")
    with open(save_file, 'rb') as f:
        masks_dictionary = pickle.load(f)

    # ⬇️ Deleted the hardcoded img_h, img_w here ⬇️

    base_dir_img = r"D:\Lidar and Radar\data_object_image_2\training\image_2"
    base_dir_calib = r"D:\Lidar and Radar\data_object_calib\training\calib"
    base_dir_lidar = r"D:\Lidar and Radar\data_object_velodyne\training\velodyne"

    output_folder = "output_projections"
    os.makedirs(output_folder, exist_ok=True)

    image_paths = sorted(Path(base_dir_img).glob("*.png"))

    for img_path in image_paths:
        frame_id = img_path.stem
        print(f"--- Processing Frame {frame_id} ---")

        calib_file_path = os.path.join(base_dir_calib, f"{frame_id}.txt")
        lidar_file_path = os.path.join(base_dir_lidar, f"{frame_id}.bin")
        image_file_path = str(img_path)

        if not os.path.exists(calib_file_path) or not os.path.exists(lidar_file_path):
            print(f"  Skipping {frame_id}: Missing calib or lidar file.")
            continue

        # ⬇️ THE FIX: Load the image early to get the exact dynamic dimensions ⬇️
        img = cv2.imread(image_file_path)
        if img is None:
            print(f"  Skipping {frame_id}: Could not read image file.")
            continue

        img_h, img_w = img.shape[:2]
        # ⬆️ ------------------------------------------------------------------ ⬆️

        car_masks = []
        for saved_path in masks_dictionary.keys():
            if saved_path.endswith(f"{frame_id}.png"):
                car_masks = masks_dictionary[saved_path]
                break

        print(f"  Found {len(car_masks)} cars from Step 1.")

        try:
            calib_data = read_kitti_calib(calib_file_path)
            points_2d, points_3d = project_lidar_to_camera(lidar_file_path, calib_data, img_w, img_h)
            print(f"  Projected {len(points_2d)} points.")
        except Exception as e:
            print(f"  Error projecting {frame_id}: {e}")
            continue

        # (Image is already loaded above, so we just calculate colors and draw)
        depths = points_3d[:,0]
        depths_normalized = np.clip(depths / 50.0, 0, 1)

        for i in range(len(points_2d)):
            u, v = int(points_2d[i, 0]), int(points_2d[i, 1])
            r = int(255 * (1 - depths_normalized[i]))
            g = int(255 * (1 - abs(0.5 - depths_normalized[i]) * 2))
            b = int(255 * depths_normalized[i])
            cv2.circle(img, (u, v), radius=2, color=(b, g, r), thickness=-1)

        output_filename = os.path.join(output_folder, f"lidar_projection_{frame_id}.png")
        cv2.imwrite(output_filename, img)
        print(f"  Saved visualization to {output_filename}\n")

    print(f"Completed processing {len(image_paths)} images.")