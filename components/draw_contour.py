import cv2


def draw_contour(path: str) -> cv2.Mat:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    binary_img = cv2.Canny(blur, 20, 160)
    contours, hierarchy = cv2.findContours(binary_img.copy(), cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("gray", gray)
    # cv2.imshow("blur", blur)
    # cv2.imshow("binary", binary_img)
    rings = int(len(contours) / 2)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    # cv2.imshow("conts", img)
    return rings, img


if __name__ == "__main__":
    path = "./Dataset_OpenCvDl_Hw2/Q1_Image/img2.jpg"
    rings, result = draw_contour(path=path)
    cv2.imshow(f"contours1 {rings}", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()