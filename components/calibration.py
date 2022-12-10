import cv2
import numpy as np
import glob
import os

from typing import Union


class CameraCalibration:
    def __init__(self) -> None:
        pass

    def imread(self, path) -> list:
        return cv2.imread(path)

    def calibration(self, images: list, w: int = 11, h: int = 8) -> list:
        """
        w, h represent chessboard width and height
        """
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((h * w, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        results = []
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.imgpoints.append(corners2)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (w, h), corners2, ret)
                results.append(img)

        # calibration
        # Now that we have our object points and image points, we are ready to go for calibration.
        # We can use the function, cv.calibrateCamera()
        # which returns the camera matrix, distortion coefficients, rotation and translation vectors etc.
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None
        )
        return results

    def undistortion(self, images: list, mtx: list, dist: list) -> list:
        results = []
        for img in images:
            img = cv2.imread(img)
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 0, (w, h)
            )
            # undistort
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y : y + h, x : x + w]
            results.append(dst)
        return results

    def hconcat(self, img1: list, img2: list) -> list:
        img2 = cv2.resize(
            img2, (img1.shape[0], img1.shape[1]), interpolation=cv2.INTER_AREA
        )
        return cv2.hconcat([img1, img2])

    def projectOnBoard(
        self,
        img: list,
        words: str,
        dir: str,
        rvecs: list,
        tvecs: list,
        mtx: list,
        dist: list,
    ) -> list:
        # library of characters
        lib = [
            os.path.join(
                "Dataset_OpenCvDl_Hw2",
                "Q3_Image",
                "Q2_lib",
                "alphabet_lib_onboard.txt",
            ),
            os.path.join(
                "Dataset_OpenCvDl_Hw2",
                "Q3_Image",
                "Q2_lib",
                "alphabet_lib_vertical.txt",
            ),
        ]
        if dir == "h":
            # read library
            fs = cv2.FileStorage(lib[0], cv2.FileStorage_READ)
            # for every char in input words
            for index, char in enumerate(words):
                # get the object points of char
                ch = fs.getNode(char).mat()
                # for every line in a char
                for line in ch:
                    x_bias = index % 3 * 3
                    y_bias = index // 3 * 3
                    line = line + [7 - x_bias, 5 - y_bias, 0]
                    # get img points
                    imagePoints, _ = cv2.projectPoints(
                        np.array(line, dtype=float), rvecs, tvecs, mtx, dist
                    )
                    imagePoints = imagePoints.reshape([-1, 2]).astype(int)
                    # draw on img
                    cv2.line(img, imagePoints[0], imagePoints[1], (0, 0, 255), 10)
            return img
        elif dir == "v":
            # read library
            fs = cv2.FileStorage(lib[1], cv2.FileStorage_READ)
            # for every char in input words
            for index, char in enumerate(words):
                # get the object points of char
                ch = fs.getNode(char).mat()
                # for every line in a char
                for line in ch:
                    x_bias = index % 3 * 3
                    y_bias = index // 3 * 3
                    line = line + [7 - x_bias, 5 - y_bias, 0]
                    # get img points
                    imagePoints, _ = cv2.projectPoints(
                        np.array(line, dtype=float), rvecs, tvecs, mtx, dist
                    )
                    imagePoints = imagePoints.reshape([-1, 2]).astype(int)
                    # draw on img
                    cv2.line(img, imagePoints[0], imagePoints[1], (0, 0, 255), 10)
            return img
        else:
            print(
                "Value Error: Expect 'h' or 'v' to project words on board horizontally or vertically."
            )

    def getRT(self, which: str) -> Union[list, None]:
        if which == "r":
            return self.rvecs
        elif which == "t":
            return self.tvecs
        else:
            print("Value Error: Expect 'r' or 't' for rvecs or tvecs")
            return None

    def getDist(self) -> list:
        return self.dist

    def getIntrinsic(self) -> list:
        return self.mtx

    def getExtrinsic(self, index: int) -> list:
        rv = self.rvecs[index]
        tv = self.tvecs[index]
        R, _ = cv2.Rodrigues(rv)
        ex = np.zeros(shape=(3, 4))
        for i in range(3):
            for j in range(4):
                if j != 3:
                    ex[i][j] = R[i][j]
                else:
                    ex[i][j] = tv[i]
        return ex

    def imshow(self, images: list, name: str, second: int) -> None:
        for img in images:
            cv2.imshow(name, img)
            cv2.waitKey(int(second * 1000))
        cv2.destroyAllWindows()


if __name__ == "__main__":
    images = glob.glob(os.path.join("Dataset_OpenCvDl_Hw2", "Q3_Image", "*.bmp"))

    model = CameraCalibration()
    ret = model.calibration(images=images)
    # model.imshow(ret, "input", 0.5)

    objpoints = model.objpoints
    corner2 = model.imgpoints
    rvecs = model.getRT("r")
    tvecs = model.getRT("t")
    mtx = model.getIntrinsic()
    dist = model.getDist()
    ret = model.undistortion(images, mtx, dist)
    # model.imshow(ret, "undistortion", 0.5)

    """Show words on board"""
    words = "ABCDEF"
    results = []
    for index, img in enumerate(ret):
        ret = model.projectOnBoard(
            img, words, "v", rvecs[index], tvecs[index], mtx, dist
        )
        results.append(ret)
    model.imshow(results, "test", 1)
