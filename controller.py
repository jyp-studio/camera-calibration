from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QFileDialog
from gui import Ui_MainWindow

import glob
import os

from components.draw_contour import draw_contour
from components.calibration import *
from components.stereo_map import *


class Controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

        self.model = CameraCalibration()

    def setup_control(self):
        self.ui.loadFolder.clicked.connect(self.open_folder)
        self.ui.loadImageL.clicked.connect(self.open_fileL)
        self.ui.loadImageR.clicked.connect(self.open_fileR)
        self.ui.btn1_1.clicked.connect(self.btn_1_1)
        self.ui.btn1_2.clicked.connect(self.btn_1_2)
        self.ui.btn2_1.clicked.connect(self.btn_2_1)
        self.ui.btn2_2.clicked.connect(self.btn_2_2)
        self.ui.btn2_3.clicked.connect(self.btn_2_3)
        self.ui.btn2_4.clicked.connect(self.btn_2_4)
        self.ui.btn2_5.clicked.connect(self.btn_2_5)
        self.ui.btn3_1.clicked.connect(self.btn_3_1)
        self.ui.btn3_2.clicked.connect(self.btn_3_2)
        self.ui.btn4_1.clicked.connect(self.btn_4_1)

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./")
        self.ui.labelLoadFolder.setText(folder_path)
        self.folder_path = folder_path

    def open_fileL(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.ui.labelLoadImageL.setText(filename)
        self.path_L = filename

    def open_fileR(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.ui.labelLoadImageR.setText(filename)
        self.path_R = filename

    def btn_1_1(self):
        self.rings = []
        for img in glob.glob(os.path.join(self.folder_path, "*jpg")):
            name = os.path.basename(img)
            rings, result = draw_contour(path=img)
            self.rings.append(rings)
            cv2.imshow(name, result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def btn_1_2(self):
        self.ui.labelCountRings1.setText(
            f"There are {self.rings[1]:2} rings in img1.jpg"
        )
        self.ui.labelCountRings2.setText(
            f"There are {self.rings[0]:2} rings in img2.jpg"
        )

    def btn_2_1(self):
        self.images = glob.glob(os.path.join(self.folder_path, "*.bmp"))
        results = self.model.calibration(images=self.images)
        self.model.imshow(results, "input", 0.5)

    def btn_2_2(self):
        print("Intrinsic:")
        print(self.model.getIntrinsic())

    def btn_2_3(self):
        index = self.ui.comboBoxFindExtrinsic.currentIndex()
        print("Extrinsic:")
        print(self.model.getExtrinsic(index))

    def btn_2_4(self):
        print("Distortion:")
        print(self.model.getDist())

    def btn_2_5(self):
        images = self.model.undistortion(
            self.images, self.model.getIntrinsic(), self.model.getDist()
        )
        results = []
        for index, img in enumerate(self.images):
            origin = self.model.imread(img)
            results.append(self.model.hconcat(origin, images[index]))
        self.model.imshow(results, "compare", 0.5)

    def btn_3_1(self):
        self.images = glob.glob(os.path.join(self.folder_path, "*.bmp"))
        ret = self.model.calibration(images=self.images)

        words = self.ui.lineEditAR.text().upper()
        results = []
        for index, path in enumerate(self.images):
            img = self.model.imread(path)
            ret = self.model.projectOnBoard(
                img,
                words,
                "h",
                self.model.getRT("r")[index],
                self.model.getRT("t")[index],
                self.model.getIntrinsic(),
                self.model.getDist(),
            )
            results.append(ret)
        self.model.imshow(results, "show on board", 1)

    def btn_3_2(self):
        self.images = glob.glob(os.path.join(self.folder_path, "*.bmp"))
        ret = self.model.calibration(images=self.images)

        words = self.ui.lineEditAR.text().upper()
        results = []
        for index, path in enumerate(self.images):
            img = self.model.imread(path)
            ret = self.model.projectOnBoard(
                img,
                words,
                "v",
                self.model.getRT("r")[index],
                self.model.getRT("t")[index],
                self.model.getIntrinsic(),
                self.model.getDist(),
            )
            results.append(ret)
        self.model.imshow(results, "show on board", 1)

    def btn_4_1(self):
        mapper = SteroDisparityMap(self.path_L, self.path_R)
        mapper.find_disarity()
        disp_norm = mapper.normalize()
        mapper.imshow("disparity", disp_norm)
        mapper.wait()

        mapper.imshow("imgL", mapper.imgL)
        mapper.imshow("imgR_dot", mapper.imgR)
        mapper.mouseTracking()
        mapper.wait()
