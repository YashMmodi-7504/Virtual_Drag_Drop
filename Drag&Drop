# Virtual_Drag_Drop
import cv2
import math
import random
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Get screen width and height
screen_width = 1280
screen_height = 720

# Initialize hand detector
detector = HandDetector(detectionCon=0.8)

# Rectangle parameters
rect_width, rect_height = 200, 200
rect_color = (255, 0, 255)  # Rectangle color (transparent)
dark_pink = (255, 20, 147)  # Dark pink border color
alpha = 0.5  # 50% transparency for rectangles

# Class for the draggable rectangle
class DragRect():
    def __init__(self, posCenter, size=[200, 200]):
        self.posCenter = posCenter
        self.size = size
        self.dragging = False  # Whether the rectangle is being dragged

    def update(self, cursor):
        # Update position if dragging
        if self.dragging:
            self.posCenter = cursor  # Update position to the cursor

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

try:
    # Fixed initial positions for rectangles
    rectangles = [
        DragRect([screen_width // 5, screen_height // 2], [rect_width, rect_height]),  # Rectangle 1
        DragRect([screen_width // 5 + rect_width + 10, screen_height // 2], [rect_width, rect_height]),  # Rectangle 2
        DragRect([screen_width // 5 + 2 * (rect_width + 10), screen_height // 2], [rect_width, rect_height]),  # Rectangle 3
        DragRect([screen_width // 5 + 3 * (rect_width + 10), screen_height // 2], [rect_width, rect_height])   # Rectangle 4
    ]

    while True:
        # Read frame from webcam
        success, img = cap.read()

        # Find hands in the image
        hands, img = detector.findHands(img)

        if hands:
            # Get the landmarks of the first detected hand
            lmList = hands[0]['lmList']  # List of hand landmarks

            if lmList:
                index_finger_tip = lmList[8]  # Index finger tip (landmark 8)
                middle_finger_tip = lmList[12]  # Middle finger tip (landmark 12)

                # Calculate the distance between index and middle finger
                dist = calculate_distance(index_finger_tip, middle_finger_tip)

                # Debugging the distance
                print(f"Distance between index and middle finger: {dist}")

                cursor = index_finger_tip  # Use the index finger for position
                cursor_x, cursor_y = cursor[0], cursor[1]

                # Drag the selected rectangle only
                for rect in rectangles:
                    # Check if index finger is inside the rectangle (for dragging)
                    cx, cy = rect.posCenter
                    w, h = rect.size
                    if cx - w // 2 < cursor_x < cx + w // 2 and cy - h // 2 < cursor_y < cy + h // 2:
                        if dist < 50:  # Threshold set to 50 for dragging
                            rect.dragging = True
                        else:
                            rect.dragging = False

                    # Update position if dragging
                    rect.update([cursor_x, cursor_y])

                # Draw all rectangles with 50% transparency
                for rect in rectangles:
                    cx, cy = rect.posCenter
                    w, h = rect.size
                    # Create a transparent rectangle
                    overlay = img.copy()
                    cv2.rectangle(overlay, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), rect_color, -1)
                    # Blend the transparent rectangle with the original frame
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                    # Draw dark pink border around the rectangle
                    cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), dark_pink, 3)

                # Draw the line between index and middle finger on the image
                cv2.line(img, (index_finger_tip[0], index_finger_tip[1]),
                         (middle_finger_tip[0], middle_finger_tip[1]), (0, 255, 0), 2)

                # Display the distance as text on the image
                cv2.putText(img, f"Distance: {int(dist)}", (int((index_finger_tip[0] + middle_finger_tip[0]) / 2),
                                                           int((index_finger_tip[1] + middle_finger_tip[1]) / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the image with transparent rectangles
        cv2.imshow("Webcam Feed with Transparent Rectangles", img)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("\nProgram interrupted by user. Exiting gracefully...")

finally:
    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close any OpenCV windows
