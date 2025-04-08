import argparse
import json
import os
import cv2
import numpy as np
import yaml

def load_images_and_poses(images_dir, poses_json):
    with open(poses_json, 'r') as f:
        poses_data = json.load(f)
    images, poses = [], []
    for entry in poses_data:
        img_path = os.path.join(images_dir, entry['file_path'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        images.append(img)
        poses.append(np.array(entry['transform_matrix']))
    return images, poses

def load_intrinsics(yaml_file):
    with open(yaml_file, 'r') as f:
        params = yaml.safe_load(f)
    intrinsics = params['cam0']['intrinsics']
    intrinsic_matrix = np.array([
        [intrinsics[0], 0, intrinsics[2]],
        [0, intrinsics[1], intrinsics[3]],
        [0, 0, 1]
    ])
    return intrinsic_matrix

def project_to_plane(img, pose, intrinsic, depth, canvas, count_canvas, canvas_origin, resolution):
    h, w = img.shape[:2]
    y_idxs, x_idxs = np.indices((h, w))
    pixels_homo = np.vstack((x_idxs.flatten(), y_idxs.flatten(), np.ones(x_idxs.size)))
    
    cam_coords = np.linalg.inv(intrinsic) @ pixels_homo * depth
    cam_coords = np.vstack((cam_coords, np.ones(cam_coords.shape[1])))

    world_coords = pose @ cam_coords
    x_world, y_world = world_coords[0], world_coords[1]

    x_world = (x_world - canvas_origin[0]) / resolution
    y_world = (y_world - canvas_origin[1]) / resolution
    x_world, y_world = x_world.astype(int), y_world.astype(int)

    valid_idx = (x_world >= 0) & (y_world >= 0) & (x_world < canvas.shape[1]) & (y_world < canvas.shape[0])
    canvas[y_world[valid_idx], x_world[valid_idx]] += img.reshape(-1, 3)[valid_idx]
    count_canvas[y_world[valid_idx], x_world[valid_idx]] += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch aerial images into orthophoto")
    parser.add_argument("--images_dir", required=True, help="Directory containing images")
    parser.add_argument("--poses_json", required=True, help="JSON file containing camera poses")
    parser.add_argument("--intrinsics_yaml", required=True, help="YAML file containing camera intrinsics")
    parser.add_argument("--depth", type=float, required=True, help="Fixed depth for projecting pixels")
    parser.add_argument("--resolution", type=float, default=0.1, help="Resolution of orthophoto (m/pixel)")
    parser.add_argument("--output", default="orthophoto.png", help="Output orthophoto file")
    args = parser.parse_args()

    intrinsic = load_intrinsics(args.intrinsics_yaml)
    images, poses = load_images_and_poses(args.images_dir, args.poses_json)

    all_coords = np.array([pose[:2, 3] for pose in poses])
    min_coords, max_coords = np.min(all_coords, axis=0), np.max(all_coords, axis=0)
    canvas_origin = min_coords - 50
    canvas_size = ((max_coords - min_coords + 100) / args.resolution).astype(int)

    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float32)
    count_canvas = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.float32)

    for img, pose in zip(images, poses):
        project_to_plane(img, pose, intrinsic, args.depth, canvas, count_canvas, canvas_origin, args.resolution)

    mask = count_canvas > 0
    canvas[mask] /= count_canvas[mask, None]
    orthophoto = np.clip(canvas, 0, 255).astype(np.uint8)

    cv2.imwrite(args.output, orthophoto)
    print(f"Orthophoto saved to {args.output}")
