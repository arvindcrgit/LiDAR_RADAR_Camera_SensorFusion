import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_sensor(directory, label):
    """
    Extracts distance and intensity metrics from a sensor's CSV directory.
    Calculates Euclidean distance for every point cloud coordinate (x, y, z).
    """
    distances = []
    intensities = []

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return None, None, None

    # Iterate and sort files to keep sequence consistent
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".csv"):
            try:
                # Load data from the CSV file
                data = np.loadtxt(os.path.join(directory, filename), delimiter=None)
                if data.ndim == 1:
                    data = data.reshape(1, -1)

                # Distance calculation: sqrt(x^2 + y^2 + z^2)
                dist = np.linalg.norm(data[:, :3], axis=1)
                distances.extend(dist)

                # Extract intensity data from column index 3
                intensities.extend(data[:, 3])
            except (ValueError, IOError, IndexError):
                continue

    d_arr = np.array(distances)
    i_arr = np.array(intensities)

    # Calculate Evaluation Metrics for Task 1
    metrics = {
        "label": label,
        "mean_int": np.mean(i_arr) if len(i_arr) > 0 else 0.0,
        "max_range": np.max(d_arr) if len(d_arr) > 0 else 0.0,
        "near_noise": np.sum(d_arr < 2.0)  # Count of points closer than 2 meters
    }
    return d_arr, i_arr, metrics


if __name__ == "__main__":
    # 1. Configuration of Dataset Root Paths (using raw strings for Windows paths)
    root_clear = r"D:\Lidar and Radar\RWUDataset\CBuilding\csv\c_building_pedestrian_clear_anon"
    root_fog = r"D:\Lidar and Radar\RWUDataset\CBuilding\csv\c_building_pedestrian_fog_anon"

    sensors = ["velodyne", "blickfeld", "radar"]
    all_results = {}

    # 2. Batch process data execution across all configurations
    print("Extracting metrics and distances from files... Please wait.")
    for s in sensors:
        all_results[f"{s}_clear"] = analyze_sensor(os.path.join(root_clear, s), f"{s.upper()} Clear")
        all_results[f"{s}_fog"] = analyze_sensor(os.path.join(root_fog, s), f"{s.upper()} Fog")

    # 3. Print Comprehensive Evaluation Metrics Table
    print("\n" + "=" * 70)
    print(f"{'Sensor Config':<20} | {'Mean Int.':<10} | {'Max Range':<10} | {'Near Noise'}")
    print("-" * 70)
    for key in sorted(all_results.keys()):
        res = all_results[key]
        if res[2] is not None:
            m = res[2]
            print(f"{m['label']:<20} | {m['mean_int']:<10.2f} | {m['max_range']:<10.2f} | {m['near_noise']}")
    print("=" * 70 + "\n")

    # 4. Generate Three Separate Overlapping Histograms (Clear vs Foggy)
    for s in sensors:
        clear_res = all_results[f"{s}_clear"]
        fog_res = all_results[f"{s}_fog"]

        # Verify both data streams populated successfully before rendering plots
        if clear_res[0] is not None and fog_res[0] is not None:
            plt.figure(figsize=(10, 5.5))

            # Draw overlaying histograms with requested colors: green for clear, yellow for fog
            plt.hist(clear_res[0], bins=100, alpha=0.5, label='Without Fog (Clear Scene)', color='green', range=(0, 30))
            plt.hist(fog_res[0], bins=100, alpha=0.5, label='With Fog (Foggy Scene)', color='yellow', range=(0, 30))

            # Label plots dynamically based on the current looping sensor
            plt.title(f'Task 1: {s.upper()} Distance Distribution Analysis')
            plt.xlabel('Distance from Sensor Origin (meters)')
            plt.ylabel('Frequency (Detections Count)')
            plt.legend(loc='upper right')
            plt.grid(axis='y', linestyle='--', alpha=0.3)

            plt.tight_layout()
            plt.show()