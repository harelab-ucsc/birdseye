import cv2
import os
import sys
import requests
import pandas as pd
import numpy as np


# Load existing clicks from manual.txt
def load_manual_clicks(filepath):
    clicks = {}
    if not os.path.exists(filepath):
        return clicks
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                frame_idx = int(parts[0])
                coords = parts[2].split(',')
                x, y = float(coords[0]), float(coords[1])
            except Exception:
                continue
            if frame_idx not in clicks:
                clicks[frame_idx] = []
            clicks[frame_idx].append((x, y))
    return clicks


def fetch_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_data = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Failed to decode image from {url}")
        return img
    except Exception as e:
        print(f"Failed to load image from {url}: {e}")
        return None


def draw_clicks(img, clicks):
    for (x, y) in clicks:
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
    return img


def draw_overlay_text(img, frame_index):
    overlay = img.copy()
    h, w = img.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    line_height = 35

    top_line = (f"Frame {frame_index + 1}/{len(image_urls)}    "
                "Left Click: Add Point    Right Click: Remove Point")
    bottom_line = "'a' = Prev Frame    'd' = Next Frame    ESC = Quit"

    top_size, _ = cv2.getTextSize(top_line, font, font_scale, thickness)
    bottom_size, _ = cv2.getTextSize(bottom_line, font, font_scale, thickness)

    top_box_height = line_height + 20
    cv2.rectangle(overlay, (5, 5), (top_size[0] + 20, 5 + top_box_height), (0, 0, 0), -1)
    cv2.putText(overlay, top_line, (10, 5 + line_height), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    bottom_box_height = line_height + 20
    y_start = h - bottom_box_height - 5
    cv2.rectangle(overlay, (5, y_start), (bottom_size[0] + 20, y_start + bottom_box_height), (0, 0, 0), -1)
    cv2.putText(overlay, bottom_line, (10, y_start + line_height), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    return img


def save_all_clicks():
    with open(output_file, 'w') as f:
        for idx, url in enumerate(image_urls):
            if idx in clicks_dict:
                for (x, y) in clicks_dict[idx]:
                    f.write(f"{idx} {url} {x},{y},1.0\n")


def mouse_callback(event, x, y, flags, param):
    global clicks_dict
    if frame_index not in clicks_dict:
        clicks_dict[frame_index] = []

    if event == cv2.EVENT_LBUTTONDOWN:
        clicks_dict[frame_index].append((x, y))
        save_all_clicks()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if clicks_dict[frame_index]:
            closest_idx = min(
                range(len(clicks_dict[frame_index])),
                key=lambda i: (clicks_dict[frame_index][i][0] - x) ** 2 +
                              (clicks_dict[frame_index][i][1] - y) ** 2
            )
            del clicks_dict[frame_index][closest_idx]
            save_all_clicks()


def show_image():
    url = image_urls[frame_index]
    img = image_cache.get(url)
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
    display_img = img.copy()

    if frame_index in clicks_dict:
        display_img = draw_clicks(display_img, clicks_dict[frame_index])

    display_img = draw_overlay_text(display_img, frame_index)
    cv2.imshow("Image", display_img)
    return True


if __name__ == "__main__":

    output_file = 'manual.txt'

    if len(sys.argv) < 2:
        print("Usage: python click_capture.py /path/to/image_urls.csv")
        sys.exit(1)

    csv_path = os.path.expanduser(sys.argv[1])
    try:
        df = pd.read_csv(csv_path)
        if 'image_url' not in df.columns:
            raise ValueError("CSV must contain a 'image_url' column.")
        image_urls = df['image_url'].tolist()
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        sys.exit(1)

    print(f"Loaded {len(image_urls)} image URLs from {csv_path}")
    if not image_urls:
        print("No URLs found. Exiting.")
        sys.exit(1)

    clicks_dict = load_manual_clicks(output_file)
    frame_index = 0

    # Preload all images into cache
    image_cache = {}

    print("Preloading all images. This may take a while...")
    for idx, url in enumerate(image_urls):
        print(f"Loading {idx+1}/{len(image_urls)}: {url}")
        img = fetch_image_from_url(url)
        if img is not None:
            image_cache[url] = img
        else:
            # Black placeholder if load fails
            image_cache[url] = np.zeros((480, 640, 3), dtype=np.uint8)
    print("All images loaded, launching GUI...")

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Image", mouse_callback)
    cv2.resizeWindow("Image", 960, 600)

    while True:
        success = show_image()
        if not success:
            key = cv2.waitKey(100)
            continue

        key = cv2.waitKey(30) & 0xFF

        if key == ord('d') and frame_index < len(image_urls) - 1:
            frame_index += 1
        elif key == ord('a') and frame_index > 0:
            frame_index -= 1
        elif key == 27:  # ESC key
            break

    cv2.destroyAllWindows()
