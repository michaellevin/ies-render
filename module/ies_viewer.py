import sys
import os
import math

from qtpy.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QTreeView,
    QFileSystemModel,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QMessageBox,
)
from qtpy.QtCore import Qt, QDir, QRectF, QSettings
from qtpy.QtGui import QImage, QPixmap, QPainter

try:
    from .ies_gen import IES_Thumbnail_Generator
except ImportError:
    from ies_gen import IES_Thumbnail_Generator


class ZoomableGraphicsView(QGraphicsView):
    WIDTH, HEIGHT = 512, 512

    def __init__(self, parent=None):
        super(ZoomableGraphicsView, self).__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self._zoom = 0
        self._empty = True
        self._scene_limits = QRectF()
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setFixedSize(self.WIDTH, self.HEIGHT)

    def set_pixmap(self, pixmap):
        self.scene.clear()
        self._zoom = 0
        item = QGraphicsPixmapItem(pixmap)
        item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(item)
        self._scene_limits = QRectF(pixmap.rect())
        viewer_size = self.WIDTH
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        self.setSceneRect(0, 0, viewer_size, viewer_size)

        # Center the pixmap item within the viewer
        x = (viewer_size - pixmap_width) / 2
        y = (viewer_size - pixmap_height) / 2
        item.setPos(x, y)

        # Set the initial view based on the image size
        if pixmap_width > viewer_size or pixmap_height > viewer_size:
            self.fitInView(self._scene_limits, Qt.KeepAspectRatio)
        else:
            self._zoom = int(
                math.log(viewer_size / max(pixmap_width, pixmap_height), 1.25)
            )

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            angle = event.angleDelta().y()
            if angle > 0:  # Zooming in
                factor = 1.25
                self._zoom += 1
                self.scale(factor, factor)
            else:  # Zooming out
                factor = 0.8
                pixmap_width = self._scene_limits.width()
                pixmap_height = self._scene_limits.height()
                min_scale_factor = max(
                    pixmap_width / self.WIDTH, pixmap_height / self.HEIGHT, 1
                )

                # Calculate the resulting scale factor if the zoom out is applied
                transform = self.transform()
                resulting_scale_factor = (
                    transform.m11() * factor
                )  # m11() is the horizontal scale factor

                # Only allow further zooming out if the resulting scale factor is greater than the minimum scale factor
                if resulting_scale_factor >= min_scale_factor:
                    self._zoom -= 1
                    self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        elif event.button() == Qt.RightButton:
            self.resetTransform()
            self._zoom = 0
            pixmap_width = self._scene_limits.width()
            pixmap_height = self._scene_limits.height()
            if pixmap_width > self.WIDTH or pixmap_height > self.HEIGHT:
                self.fitInView(self._scene_limits, Qt.KeepAspectRatio)
            else:
                # For smaller images, set the view to the original image size
                x = (self.WIDTH - pixmap_width) / 2
                y = (self.HEIGHT - pixmap_height) / 2
                self.scene.items()[0].setPos(x, y)
                print(self.scene.items()[0].pos())
        super(ZoomableGraphicsView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setDragMode(QGraphicsView.NoDrag)
        super(ZoomableGraphicsView, self).mouseReleaseEvent(event)


class IES_Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IES Viewer")
        self.setObjectName("IES Viewer")
        # Initialize UI elements
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()

        # IES Files TreeView
        self.file_system_model = QFileSystemModel()
        self.file_system_model.setRootPath(QDir.rootPath())
        self.file_system_model.setNameFilters(["*.ies"])
        self.file_system_model.setNameFilterDisables(False)
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_system_model)
        self.tree_view.setRootIndex(self.file_system_model.index("examples"))
        self.tree_view.hideColumn(1)  # Hide "Size" column
        self.tree_view.hideColumn(2)  # Hide "Type" column
        self.tree_view.hideColumn(3)  # Hide "Date Modified" column
        main_layout.addWidget(self.tree_view)

        # Image Label
        self.graphics_view = ZoomableGraphicsView()
        main_layout.addWidget(self.graphics_view)

        # Settings and Buttons Layout
        settings_layout = QVBoxLayout()

        # Render Size ComboBox with Label
        settings_layout.addWidget(QLabel("Render Size:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["16", "32", "64", "128", "256", "512", "1024"])
        settings_layout.addWidget(self.size_combo)

        # Horizontal Angle DoubleSpinBox with Label
        settings_layout.addWidget(QLabel("Horizontal Angle:"))
        self.horizontal_angle_spinbox = QDoubleSpinBox()
        settings_layout.addWidget(self.horizontal_angle_spinbox)

        # Distance DoubleSpinBox with Label
        settings_layout.addWidget(QLabel("Distance from the wall:"))
        self.distance_spinbox = QDoubleSpinBox()
        self.distance_spinbox.setSingleStep(0.01)
        settings_layout.addWidget(self.distance_spinbox)

        # Blur CheckBox with Label
        settings_layout.addWidget(QLabel("Add Blur:"))
        self.blur_checkbox = QCheckBox()
        settings_layout.addWidget(self.blur_checkbox)

        # Blur Radius DoubleSpinBox with Label
        settings_layout.addWidget(QLabel("Blur Radius:"))
        self.blur_radius_spinbox = QDoubleSpinBox()
        self.blur_radius_spinbox.setRange(0, 10)
        self.blur_radius_spinbox.setSingleStep(0.5)
        settings_layout.addWidget(self.blur_radius_spinbox)

        settings_layout.addStretch(1)

        # Generate Button
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_image)
        self.generate_button.setStyleSheet(
            "background-color: #4CAF50; color: white; border: none; padding: 10px; text-align: center; text-decoration: none; font-size: 16px; margin: 4px 2px; border-radius: 12px;"
        )
        settings_layout.addWidget(self.generate_button)

        # Save Button
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setStyleSheet(
            "background-color: #4169E1; color: white; border: none; padding: 10px; text-align: center; text-decoration: none; font-size: 16px; margin: 4px 2px; border-radius: 12px;"
        )
        settings_layout.addWidget(self.save_button)

        main_layout.addLayout(settings_layout)
        self.setLayout(main_layout)

        self.load_settings()
        self.window().destroyed.connect(self.save_settings)

    def generate_image(self):
        # Get the selected IES file
        selected_index = self.tree_view.currentIndex()
        selected_file_path = self.file_system_model.filePath(selected_index)
        if not os.path.isfile(selected_file_path):
            print("Please select a valid IES file.")
            return

        # Get the settings values
        render_size = int(self.size_combo.currentText())
        horizontal_angle = self.horizontal_angle_spinbox.value()
        distance = self.distance_spinbox.value()
        blur = self.blur_checkbox.isChecked()
        if blur:
            blur_radius = self.blur_radius_spinbox.value()
        else:
            blur_radius = 0
        # Call your function to generate the PIL image
        self.selected_file_path = selected_file_path
        self.tb_gen = IES_Thumbnail_Generator(selected_file_path)
        pil_image = self.tb_gen.render(
            render_size, horizontal_angle, distance, blur_radius, save=False
        )

        # Convert PIL Image to QImage
        qimage = QImage(
            pil_image.tobytes("raw", "RGB"),
            pil_image.width,
            pil_image.height,
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimage)

        # Display the image
        # self.image_label.setPixmap(pixmap)
        self.graphics_view.set_pixmap(pixmap)

    def save_image(self):
        # Get the pixmap from the QGraphicsPixmapItem in the scene
        pixmap_item = self.graphics_view.scene.items()[0]
        pixmap = pixmap_item.pixmap()

        # Construct the output path based on the current settings
        out_path = self.selected_file_path.replace(
            ".ies",
            f"_s{int(self.size_combo.currentText())}_d{self.distance_spinbox.value()}_h{self.horizontal_angle_spinbox.value()}.png",
        )

        # Save the pixmap
        pixmap.save(out_path)
        QMessageBox.information(self, "Saved", f"Saved to {out_path}")

    def load_settings(self):
        settings = QSettings("lz", "IES_Viewer")

        # Load settings and set the values of the UI elements
        self.size_combo.setCurrentText(settings.value("size", "256"))
        self.horizontal_angle_spinbox.setValue(
            float(settings.value("horizontal_angle", "0"))
        )
        self.distance_spinbox.setValue(float(settings.value("distance", "0")))
        self.blur_checkbox.setChecked(settings.value("blur", "false") == "true")

        # Load the blur radius value
        self.blur_radius_spinbox.setValue(float(settings.value("blur_radius", "0")))

    def save_settings(self):
        settings = QSettings("lz", "IES_Viewer")

        # Save the current values of the UI elements to settings
        settings.setValue("size", self.size_combo.currentText())
        settings.setValue(
            "horizontal_angle", str(self.horizontal_angle_spinbox.value())
        )
        settings.setValue("distance", str(self.distance_spinbox.value()))
        settings.setValue("blur", "true" if self.blur_checkbox.isChecked() else "false")

        # Save the blur radius value
        settings.setValue("blur_radius", str(self.blur_radius_spinbox.value()))

    def closeEvent(self, event) -> None:
        self.save_settings()
        return super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = IES_Viewer()
    viewer.show()
    sys.exit(app.exec_())
