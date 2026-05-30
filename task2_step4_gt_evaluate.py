import numpy as np
import pickle
import os
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


def read_kitti_labels(filepath):
    cars = []
    if not os.path.exists(filepath):
        return cars

    with open(filepath, 'r') as f:
        for line in f.readlines():
            data = line.split()
            if data[0] == 'Car':
                dimensions = np.array([float(data[8]), float(data[9]), float(data[10])])
                location = np.array([float(data[11]), float(data[12]), float(data[13])])
                rotation_y = float(data[14])

                cars.append({
                    'dimensions': dimensions,
                    'location': location,
                    'rotation_y': rotation_y,
                    'true_distance': location[2]  # In KITTI Ground Truth, Z is the distance to the center
                })
    return cars


def project_lidar_to_camera_3d(velo_path, calib, img_width, img_height):
    velo_points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    points_3d = velo_points[:, :3]
    ones = np.ones((points_3d.shape[0], 1))
    points_3d_homo = np.hstack([points_3d, ones])

    Tr_velo_to_cam = np.vstack([calib['Tr_velo_to_cam'], [0, 0, 0, 1]])
    R0_rect = np.eye(4)
    R0_rect[:3, :3] = calib['R0_rect']
    P2 = calib['P2']

    # Transform to Camera 3D coordinates (X=Right, Y=Down, Z=Forward)
    cam_3d_homo = (R0_rect @ Tr_velo_to_cam) @ points_3d_homo.T
    cam_3d_points = cam_3d_homo[:3, :].T

    # Project to 2D image pixels
    proj_matrix = P2 @ R0_rect @ Tr_velo_to_cam
    points_2d_homo = proj_matrix @ points_3d_homo.T

    depths = points_2d_homo[2, :]
    valid_depth_mask = depths > 0

    u = points_2d_homo[0, :] / depths
    v = points_2d_homo[1, :] / depths

    valid_uv_mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    valid_mask = valid_depth_mask & valid_uv_mask

    projected_2d = np.vstack((u[valid_mask], v[valid_mask])).T
    valid_cam_3d = cam_3d_points[valid_mask]

    return projected_2d, valid_cam_3d


def check_points_in_box(cam_points, dimensions, location, rotation_y, tolerance=1.0):
    """Checks points against the mathematical KITTI box with a real-world sensor tolerance margin."""
    h, w, l = dimensions
    shifted = cam_points - np.array(location)

    angle = -rotation_y
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])

    local_points = shifted @ R.T

    in_box = (
            (local_points[:, 0] >= -(w / 2) - tolerance) & (local_points[:, 0] <= (w / 2) + tolerance) &
            (local_points[:, 1] >= -h - tolerance) & (local_points[:, 1] <= 0 + tolerance) &
            (local_points[:, 2] >= -(l / 2) - tolerance) & (local_points[:, 2] <= (l / 2) + tolerance)
    )
    return in_box


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

    # DIRECTORIES
    base_dir_img = r"D:\Lidar and Radar\data_object_image_2\training\image_2"
    base_dir_calib = r"D:\Lidar and Radar\data_object_calib\training\calib"
    base_dir_lidar = r"D:\Lidar and Radar\data_object_velodyne\training\velodyne"
    base_dir_label = r"D:\Lidar and Radar\data_object_label_2\training\label_2"

    output_folder = "output_evaluations"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Evaluation reports will be saved to: ./{output_folder}/")

    image_paths = sorted(Path(base_dir_img).glob("*.png"))

    for img_path in image_paths:
        frame_id = img_path.stem
        print(f"==================================================")
        print(f" PROCESSING FRAME: {frame_id}")
        print(f"==================================================")

        calib_file = os.path.join(base_dir_calib, f"{frame_id}.txt")
        lidar_file = os.path.join(base_dir_lidar, f"{frame_id}.bin")
        label_file = os.path.join(base_dir_label, f"{frame_id}.txt")

        if not os.path.exists(calib_file) or not os.path.exists(lidar_file) or not os.path.exists(label_file):
            continue

        car_masks = []
        for saved_path in masks_dictionary.keys():
            if saved_path.endswith(f"{frame_id}.png"):
                car_masks = masks_dictionary[saved_path]
                break

        gt_cars = read_kitti_labels(label_file)

        frame_log = f"Evaluation Report for Frame: {frame_id}\n"
        frame_log += f"{'-' * 50}\n"
        frame_log += f"YOLO found {len(car_masks)} cars. Ground Truth records {len(gt_cars)} cars.\n\n"

        print(f"  YOLO found {len(car_masks)} cars. Ground Truth records {len(gt_cars)} cars.")

        if len(car_masks) == 0:
            print("  Skipping evaluation (No YOLO masks to check).\n")
            continue

        img_h, img_w = car_masks[0].shape

        try:
            calib_data = read_kitti_calib(calib_file)
            points_2d, cam_3d_points = project_lidar_to_camera_3d(lidar_file, calib_data, img_w, img_h)
        except Exception as e:
            print(f"  Error projecting {frame_id}: {e}")
            continue

        u_coords = points_2d[:, 0].astype(int)
        v_coords = points_2d[:, 1].astype(int)

        for idx, mask in enumerate(car_masks):
            car_title = f"--- Evaluating YOLO Car {idx + 1} ---"
            print(f"\n  {car_title}")
            frame_log += f"{car_title}\n"

            point_is_inside_mask = mask[v_coords, u_coords] > 0.5
            car_points_3d = cam_3d_points[point_is_inside_mask]

            if len(car_points_3d) == 0:
                msg = "Skipped: 0 LiDAR points fell inside this mask. (Car too far or occluded)"
                print(f"    {msg}")
                frame_log += f"    {msg}\n\n"
                continue

            # ---------------------------------------------------------
            # 3D SPATIAL OUTLIER REMOVAL (Camera Coordinates)
            # In Camera coords: X is Width, Y is Height, Z is Depth
            # ---------------------------------------------------------
            med_x = np.median(car_points_3d[:, 0])
            med_y = np.median(car_points_3d[:, 1])
            med_z = np.median(car_points_3d[:, 2])

            spatial_mask = (
                    (np.abs(car_points_3d[:, 0] - med_x) < 1.5) &  # Width limit
                    (np.abs(car_points_3d[:, 1] - med_y) < 1.0) &  # Height limit
                    (np.abs(car_points_3d[:, 2] - med_z) < 3.0)  # Depth limit
            )
            car_points_3d = car_points_3d[spatial_mask]

            if len(car_points_3d) == 0:
                msg = "Skipped: 0 LiDAR points left after filtering noise."
                print(f"    {msg}")
                frame_log += f"    {msg}\n\n"
                continue

            # ---------------------------------------------------------
            # NEW PERCENTILE MATH
            # ---------------------------------------------------------
            dist_5th = np.percentile(car_points_3d[:, 2], 5)
            dist_10th = np.percentile(car_points_3d[:, 2], 10)
            dist_50th = np.percentile(car_points_3d[:, 2], 50)

            # Official distance uses the 5th percentile
            official_estimated_distance = dist_5th

            best_match = None
            min_dist_diff = float('inf')

            for gt_car in gt_cars:
                dist_diff = abs(official_estimated_distance - gt_car['true_distance'])
                # We allow a larger matching tolerance because our bumper measurement
                # will systematically be ~2m shorter than the ground truth center
                if dist_diff < min_dist_diff and dist_diff < 5.0:
                    min_dist_diff = dist_diff
                    best_match = gt_car

            if best_match:
                inside_mask = check_points_in_box(car_points_3d, best_match['dimensions'], best_match['location'],
                                                  best_match['rotation_y'])
                correct_points = np.sum(inside_mask)
                total_points = len(car_points_3d)
                bleed_out = total_points - correct_points
                bleed_out_percent = (bleed_out / total_points) * 100 if total_points > 0 else 0

                stats_report = (
                    f"    Matched with Ground Truth Car at {best_match['true_distance']:.2f}m (Center)\n"
                    f"    --------------------------------------------------\n"
                    f"    Distance breakdown along the car's body:\n"
                    f"      5th Percentile (Bumper):     {dist_5th:.2f}m\n"
                    f"      10th Percentile (Trunk):     {dist_10th:.2f}m\n"
                    f"      50th Percentile (Bulk Mass): {dist_50th:.2f}m\n"
                    f"    --------------------------------------------------\n"
                    f"    Systematic Offset (Error):   {min_dist_diff:.2f}m (Distance from Bumper to Center)\n"
                    f"    Total LiDAR Points:          {total_points}\n"
                    f"    Correct Points (In Box):     {correct_points}\n"
                    f"    Incorrect Bleed-out Points:  {bleed_out} ({bleed_out_percent:.1f}%)\n"
                )

                print(stats_report, end="")
                frame_log += stats_report + "\n"

            else:
                msg1 = f"Calculated Distance (5th Percentile): {official_estimated_distance:.2f}m"
                msg2 = "Could not match this YOLO mask to a Ground Truth car."
                print(f"    {msg1}\n    {msg2}")
                frame_log += f"    {msg1}\n    {msg2}\n\n"

        print("\n")

        save_log_path = os.path.join(output_folder, f"{frame_id}_evaluation_report.txt")
        with open(save_log_path, "w") as log_file:
            log_file.write(frame_log)