import cv2
import numpy as np
from tkinter import filedialog
from tkinter import Tk, Button, Label, Canvas, PhotoImage

class ImageScannerApp:
    def __init__(self, root):
        # Initialize the ImageScannerApp
        self.root = root
        self.root.title("Autonomous Image Scanner")

        # Initialize attributes to store image path and loaded images
        self.image_path = ""
        self.original_image = None
        self.scanned_image = None

        # Create GUI elements
        self.label = Label(root, text="Select an image:")
        self.label.pack(pady=10)

        self.canvas = Canvas(root, width=800, height=600)
        self.canvas.pack()

        # Button to load image
        self.load_button = Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        # Button to scan document
        self.scan_button = Button(root, text="Scan Document", command=self.scan_document)
        self.scan_button.pack(pady=10)

    def load_image(self):
        # Open file dialog to select an image
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if self.image_path:
            # Read the selected image and display it on the canvas
            self.original_image = cv2.imread(self.image_path)
            self.display_image(self.original_image)

    def display_image(self, image):
        # Display the given image on the canvas
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        photo = PhotoImage(data=cv2.imencode('.ppm', image_rgb)[1].tobytes())
        self.canvas.config(width=photo.width(), height=photo.height())
        self.canvas.create_image(0, 0, anchor='nw', image=photo)
        self.canvas.photo = photo

    def scan_document(self):
        if self.original_image is not None:
            # Perform document scan and display the result
            scanned_image = self.perform_document_scan(self.original_image)
            self.display_image(scanned_image)

    def perform_document_scan(self, image):
        # Image processing steps to detect and scan the document
        # Replace this with your own document scanning logic
        # Example: contour detection, perspective transformation, etc.

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply Canny edge detection
        edged = cv2.Canny(blurred, 30, 50)

        # Find contours in the edged image
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            p = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.01 * p, True)

            if len(approx) == 4:
                target = approx
                break

        # Map the detected corners to a new perspective
        approx = self.mapp(target)
        pts = np.float32([[0, 0], [800, 0], [800, 600], [0, 600]])
        op = cv2.getPerspectiveTransform(approx, pts)
        scanned_image = cv2.warpPerspective(image, op, (800, 600))

        return scanned_image

    def mapp(self, h):
        # Rearrange the corner points to a standard order
        h = h.reshape((4, 2))
        hnew = np.zeros((4, 2), dtype=np.float32)
        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]

        diff = np.diff(h, axis=1)
        hnew[1] = h[np.argmax(diff)]
        hnew[3] = h[np.argmax(diff)]
        return hnew

if __name__ == "__main__":
    # Create Tkinter root window and start the application
    root = Tk()
    app = ImageScannerApp(root)
    root.mainloop()