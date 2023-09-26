from qtpy.QtGui import QImage, QColor, QPixmap
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget


# class MainWindow(QMainWindow):
#     def __init__(self):
#         super(MainWindow, self).__init__()

#         self.setWindowTitle("Polar Coordinates Visualization")

#         widget = QWidget(self)
#         layout = QVBoxLayout(widget)
#         self.label = QLabel()
#         layout.addWidget(self.label)
#         self.setCentralWidget(widget)

#         # Generate QImage
#         image = QImage(512, 512, QImage.Format_ARGB32)

#         for x in range(512):
#             for y in range(512):
#                 # Calculate relative coordinates
#                 dx = x - 255.5
#                 dy = y - 255.5

#                 r = np.sqrt(dx * dx + dy * dy)
#                 theta = np.degrees(np.arctan2(-dx, dy))  # Convert to degrees
#                 # if theta < 0:  # Adjust to [0, 180] range
#                 #     theta += 180

#                 # Compute the pixel value based on decay and interpolation
#                 pixel_value = compute_pixel_value(r, math.fabs(theta))
#                 # print(pixel_value)
#                 # Set the pixel color
#                 image.setPixelColor(x, y, QColor(pixel_value, pixel_value, pixel_value))

#         # Display the image in the QLabel
#         self.label.setPixmap(QPixmap(image))
