
import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 255, 0), thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def detect_lanes(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and help with edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Define a region of interest (ROI) based on image size
    height, width = edges.shape
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    roi_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(roi_edges, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]),
                            minLineLength=40, maxLineGap=25)

    # Create a blank image to draw the lines on
    line_image = np.zeros_like(image)

    # Filter lines based on slope to separate left and right lanes
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    # Draw left and right lane lines in green
    draw_lines(line_image, left_lines, color=(0, 255, 0))
    draw_lines(line_image, right_lines, color=(0, 255, 0))

    # Combine the original frame with the lane lines
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return result, left_lines, right_lines

def get_curvature(left_lane, right_lane, height):
    # Extract x and y coordinates of left and right lane points
    left_points = np.vstack([line[:, :2] for line in left_lane])
    right_points = np.vstack([line[:, :2] for line in right_lane])

    # Fit a second-degree polynomial to the lane points
    left_fit = np.polyfit(left_points[:, 0], left_points[:, 1], 2)
    right_fit = np.polyfit(right_points[:, 0], right_points[:, 1], 2)

    # Calculate curvature for the bottom of the image
    y_eval = height - 1
    left_curvature = ((1 + (2 * left_fit[0] * y_eval + left_fit[1])**2)**1.5) / np.abs(2 * left_fit[0])
    right_curvature = ((1 + (2 * right_fit[0] * y_eval + right_fit[1])**2)**1.5) / np.abs(2 * right_fit[0])

    return left_curvature, right_curvature

# Open a video file or use 0 for webcam
cap = cv2.VideoCapture("C:\\Users\\Hoda Sherif\\Downloads\\source_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Unable to read the frame.")
        break

    # Detect lanes in the frame
    lane_detected_frame, left_lane, right_lane = detect_lanes(frame)

    # Get the height of the frame
    height, _ = frame.shape[:2]

    # Get curvature for the left and right lanes
    left_curvature, right_curvature = get_curvature(left_lane, right_lane, height)

    # Display curvature information on the frame
    cv2.putText(lane_detected_frame, f"Left Curvature: {round(left_curvature, 2)} m", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(lane_detected_frame, f"Right Curvature: {round(right_curvature, 2)} m", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the original and lane-detected frames
    cv2.imshow('Lane Detection with Curvature', lane_detected_frame)

    # Press 'ESC' key to exit the loop
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()