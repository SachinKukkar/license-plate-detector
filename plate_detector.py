import cv2
import numpy as np

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at path '{image_path}' not found.")
    return img

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    return gray, edges

def find_license_plate_regions(edges, aspect_ratio_range=(2.0, 5.0)):
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    valid_regions = []

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                valid_regions.append((x, y, w, h))
    
    return valid_regions

def draw_bounding_boxes(img, regions):
    for (x, y, w, h) in regions:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
    return img

def save_output(img, output_path="output.jpg"):
    cv2.imwrite(output_path, img)

def main(image_path, output_path="output.jpg"):
    img = load_image(image_path)
    gray, edges = preprocess_image(img)
    regions = find_license_plate_regions(edges)
    img_with_boxes = draw_bounding_boxes(img, regions)
    save_output(img_with_boxes, output_path)

if __name__ == "__main__":
    main("input.jpg")
