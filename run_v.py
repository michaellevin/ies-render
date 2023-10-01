import sys
from qtpy.QtWidgets import QApplication
from module import IES_Viewer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = IES_Viewer()
    viewer.show()
    sys.exit(app.exec_())
