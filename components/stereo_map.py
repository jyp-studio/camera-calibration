import numpy as np
import cv2
import os

# path = os.path.join("Dataset_OpenCvDl_Hw2", "Q4_Image")


class SteroDisparityMap:
    def __init__(self, path1: str, path2: str, size: tuple = (1362, 924)) -> None:
        self.imgL = cv2.imread(path1)
        self.imgR = cv2.imread(path2)
        self.imgL = cv2.resize(self.imgL, size)
        self.imgR = cv2.resize(self.imgR, size)

        self.grayL = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
        self.grayR = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
        self.grayL = cv2.resize(self.grayL, size)
        self.grayR = cv2.resize(self.grayR, size)

    def find_disarity(self) -> list:
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        self.disparity = stereo.compute(self.grayL, self.grayR)
        return self.disparity

    def normalize(self) -> list:
        self.disp_norm = cv2.normalize(
            self.disparity,
            self.disparity,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        return self.disp_norm

    def imshow(self, name: str, img: list) -> None:
        cv2.imshow(name, img)

    def wait(self) -> None:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            disp = int(self.disparity[y][x] / 16)
            if disp <= 0:
                return

            ret = self.imgR.copy()
            point = (x - disp, y)
            ret = cv2.circle(ret, point, 10, (0, 255, 0), -1)
            cv2.imshow("imgR_dot", ret)

    def mouseTracking(self) -> None:
        cv2.setMouseCallback("imgL", self.draw)


if __name__ == "__main__":
    path_L = "./Dataset_OpenCvDl_Hw2/Q4_Image/imL.png"
    path_R = "./Dataset_OpenCvDl_Hw2/Q4_Image/imR.png"
    mapper = SteroDisparityMap(path_L, path_R)
    disparity = mapper.find_disarity()
    disp_norm = mapper.normalize()
    mapper.imshow("disparity", disp_norm)
    mapper.wait()

    mapper.imshow("imgL", mapper.imgL)
    mapper.imshow("imgR_dot", mapper.imgR)
    mapper.mouseTracking()
    mapper.wait()
