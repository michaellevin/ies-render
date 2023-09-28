import sys
import os

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
)
from qtpy.QtCore import Qt, QDir, QRectF
from qtpy.QtGui import QImage, QPixmap, QPainter

try:
    from .ies_gen import IES_Thumbnail_Generator
except ImportError:
    from ies_gen import IES_Thumbnail_Generator


class ZoomableGraphicsView(QGraphicsView):
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

    def set_pixmap(self, pixmap):
        self.scene.clear()
        self._zoom = 0
        item = QGraphicsPixmapItem(pixmap)
        item.setTransformationMode(Qt.SmoothTransformation)
        self.scene.addItem(item)
        self._scene_limits = QRectF(pixmap.rect())  # Set scene size to image size.
        self.setSceneRect(self._scene_limits)
        self.fitInView(self._scene_limits, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.ControlModifier:
            angle = event.angleDelta().y()
            if angle > 0:
                factor = 1.25
                self._zoom += 1
                self.scale(factor, factor)
            else:
                factor = 0.8
                self._zoom -= 1
                if self._zoom > 0:
                    self.scale(factor, factor)
                elif self._zoom == 0:
                    self.fitInView(self._scene_limits, Qt.KeepAspectRatio)
                else:
                    self._zoom = 0
                    self.fitInView(self._scene_limits, Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        super(ZoomableGraphicsView, self).mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setDragMode(QGraphicsView.NoDrag)
        super(ZoomableGraphicsView, self).mouseReleaseEvent(event)


class IESViewer(QWidget):
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
        # self.image_label = QLabel()
        # self.image_label.setFixedSize(256, 256)
        # main_layout.addWidget(self.image_label)
        self.graphics_view = ZoomableGraphicsView()
        self.graphics_view.setFixedSize(512, 512)
        main_layout.addWidget(self.graphics_view)

        # Settings and Buttons Layout
        settings_layout = QVBoxLayout()

        # Render Size ComboBox with Label
        settings_layout.addWidget(QLabel("Render Size:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(["128", "256", "512", "1024"])
        settings_layout.addWidget(self.size_combo)

        # Horizontal Angle DoubleSpinBox with Label
        settings_layout.addWidget(QLabel("Horizontal Angle:"))
        self.horizontal_angle_spinbox = QDoubleSpinBox()
        settings_layout.addWidget(self.horizontal_angle_spinbox)

        # Distance DoubleSpinBox with Label
        settings_layout.addWidget(QLabel("Distance from the wall:"))
        self.distance_spinbox = QDoubleSpinBox()
        settings_layout.addWidget(self.distance_spinbox)

        # Blur CheckBox with Label
        settings_layout.addWidget(QLabel("Add Blur:"))
        self.blur_checkbox = QCheckBox()
        settings_layout.addWidget(self.blur_checkbox)

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

        # Call your function to generate the PIL image
        self.selected_file_path = selected_file_path
        self.tb_gen = IES_Thumbnail_Generator(selected_file_path)
        pil_image = self.tb_gen.render(
            render_size, horizontal_angle, distance, blur, save=False
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
        out_path = self.selected_file_path.replace(
            ".ies",
            f"_s{int(self.size_combo.currentText())}_d{self.distance_spinbox.value()}_h{self.horizontal_angle_spinbox.value()}.png",
        )
        self.image_label.pixmap().save(out_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = IESViewer()
    viewer.show()
    sys.exit(app.exec_())
