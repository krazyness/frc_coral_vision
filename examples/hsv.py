import cv2
import numpy as np
import os
import glob

def nothing(x):
    pass

def detect_vertical_lines(mask, sobel_thresh, min_line_length, max_line_gap):
    # Apply Sobel edge detection to emphasize vertical edges
    sobel_x = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)
    sobel_x = np.uint8(255 * sobel_x / np.max(sobel_x))

    # Apply threshold to create a binary edge map
    _, edges = cv2.threshold(sobel_x, sobel_thresh, 255, cv2.THRESH_BINARY)

    # Morphological transformation to close gaps
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)

    detected_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:  
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) > 5:  # Filter near-vertical lines
                    detected_lines.append((x1, y1, x2, y2))

    return detected_lines

# Load images from folder
image_folder = "images"  
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.*")))[:6]  

if not image_paths:
    print("Error: No images found in folder.")
    exit()

# Load and resize images
max_grid_width = 800
max_single_width = 500  # Max width per image
images = []

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        continue

    # Resize keeping aspect ratio
    height, width = img.shape[:2]
    if width > max_single_width:
        scale = max_single_width / width
        new_width = max_single_width
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    images.append(img)

if not images:
    print("Error: No valid images found.")
    exit()

# Arrange images in a grid (max 3 per row)
rows = [images[i:i + 3] for i in range(0, len(images), 3)]

# Make all images the same height for proper grid stacking
max_height = max(img.shape[0] for row in rows for img in row)

# Pad images to the same height
for i, row in enumerate(rows):
    for j in range(len(row)):
        h, w = row[j].shape[:2]
        if h < max_height:
            pad = np.zeros((max_height - h, w, 3), dtype=np.uint8)
            row[j] = np.vstack((row[j], pad))

# Stack images into a single image grid
grid_images = [np.hstack(row) for row in rows]
final_image = np.vstack(grid_images)

# Ensure total width doesn't exceed max_grid_width
height, width = final_image.shape[:2]
if width > max_grid_width:
    scale = max_grid_width / width
    new_width = max_grid_width
    new_height = int(height * scale)
    final_image = cv2.resize(final_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Convert to HSV
hsv_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2HSV)

# Create trackbars
cv2.namedWindow("Trackbars")

cv2.createTrackbar("Lower H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("Lower S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Lower V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Upper H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Upper V", "Trackbars", 255, 255, nothing)

cv2.createTrackbar("Sobel Thresh", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("Min Line Length", "Trackbars", 50, 200, nothing)
cv2.createTrackbar("Max Line Gap", "Trackbars", 10, 50, nothing)

while True:
    lower_h = cv2.getTrackbarPos("Lower H", "Trackbars")
    lower_s = cv2.getTrackbarPos("Lower S", "Trackbars")
    lower_v = cv2.getTrackbarPos("Lower V", "Trackbars")
    upper_h = cv2.getTrackbarPos("Upper H", "Trackbars")
    upper_s = cv2.getTrackbarPos("Upper S", "Trackbars")
    upper_v = cv2.getTrackbarPos("Upper V", "Trackbars")

    sobel_thresh = cv2.getTrackbarPos("Sobel Thresh", "Trackbars")
    min_line_length = cv2.getTrackbarPos("Min Line Length", "Trackbars")
    max_line_gap = cv2.getTrackbarPos("Max Line Gap", "Trackbars")

    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    result = cv2.bitwise_and(final_image, final_image, mask=mask)

    all_lines = detect_vertical_lines(mask, sobel_thresh, min_line_length, max_line_gap)

    output_image = result.copy()
    original_lines_image = final_image.copy()

    # Draw all detected vertical lines in green
    for (x1, y1, x2, y2) in all_lines:
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(original_lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Filtered Output", output_image)
    cv2.imshow("Original Image with Lines", original_lines_image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
