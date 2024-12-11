import cv2
import numpy as np
import yaml
import argparse
import cv2.aruco as aruco

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Draw confidence ellipses around an AprilTag with visual emphasis on precision and confidence levels.")
parser.add_argument('config_path', type=str, help="Path to the YAML configuration file.")
args = parser.parse_args()

# Load settings from the YAML file
with open(args.config_path, 'r') as file:
    config = yaml.safe_load(file)

# Parameters from YAML
std_dev_x = config['apriltag']['std_dev_x']
std_dev_y = config['apriltag']['std_dev_y']
transparency = config['apriltag']['transparency']
image_path = config['apriltag']['image_path']

# Load image
image = cv2.imread(image_path)
output = image.copy()

# Check for blurriness
def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

laplacian_variance = is_blurry(image)
#print(laplacian_variance)
if laplacian_variance < 1000:  # Threshold value for variance might need adjustment
    print("Warning: The image is blurry.")

# Detect AprilTag
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
# Remove explicit parameters creation to use defaults
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)

# Check if at least one tag was detected
if ids is not None and len(corners) > 0:
    # Calculate the center of the first detected tag
    c = corners[0][0]
    center = (int(c[:, 0].mean()), int(c[:, 1].mean()))

    # Draw ellipses with different colors and transparencies
    scales = [(1.96, (255, 0, 0), "Blue"), (1, (0, 0, 255), "Red")]  # Two ellipses
    for scale, color, description in scales:
        axes_lengths = (int(scale * std_dev_x), int(scale * std_dev_y))
        overlay = output.copy()
        cv2.ellipse(overlay, center, axes_lengths, 0, 0, 360, color, -1)
        cv2.addWeighted(overlay, transparency, output, 1 - transparency, 0, output)

# Show image
window_name = 'AprilTag with Confidence Ellipses'
cv2.imshow(window_name, output)

# Wait for the window to be closed
while True:
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break
    cv2.waitKey(100)

cv2.destroyAllWindows()
