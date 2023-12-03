import cv2
import numpy as np

def filter_color(image, lower_range, upper_range):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask using the specified color range
    mask = cv2.inRange(hsv, lower_range, upper_range)
    
    return mask

def find_contours(mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def detect_shapes(contours):
    detected_shapes = []
    
    for contour in contours:
        # Skip small contours
        if cv2.contourArea(contour) < 100:
            continue
        
        # Approximate the contour to get the shape
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Classify the shape based on the number of vertices
        vertices = len(approx)
        if vertices == 3:
            shape_type = "Triangle"
        elif vertices == 4:
            # Check if it's a square or rectangle based on aspect ratio
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            shape_type = "Square" if 0.9 <= aspect_ratio <= 1.1 else "Rectangle"
        elif vertices >= 8:
            shape_type = "Circle"
        else:
            shape_type = "Unknown"
        
        detected_shapes.append((contour, shape_type))
    
    return detected_shapes

def draw_shapes(image, shapes):
    for contour, shape_type in shapes:
        # Draw the contours and shape type on the image
        cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)
        cv2.putText(image, shape_type, (contour[0][0][0], contour[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Open the video capture (use 0 for default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read the frame.")
        break

    # Define the color range for detection (adjust based on the color you want to detect)
    lower_color = np.array([30, 50, 50])
    upper_color = np.array([90, 255, 255])

    # Filter the specified color
    color_mask = filter_color(frame, lower_color, upper_color)

    # Find contours in the filtered color mask
    contours = find_contours(color_mask)

    # Detect and classify shapes
    detected_shapes = detect_shapes(contours)

    # Draw the detected shapes on the original frame
    draw_shapes(frame, detected_shapes)

    # Display the original and processed frames
    cv2.imshow('Original Frame', frame)

    # Press 'ESC' key to exit the loop
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()