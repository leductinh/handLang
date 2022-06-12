import cv2
import numpy as np
import time

# Cac khai bao bien
bgModel = None

# Ham xoa nen khoi anh
def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Khai bao kich thuoc vung detection region
cap_region_x_begin = 0.5
cap_region_y_end = 0.8

# Cac thong so lay threshold
threshold = 60
blurValue = 41
bgSubThreshold = 50  # 50
learningRate = 0

# Nguong du doan ky tu
predThreshold = 95

isBgCaptured = 0  # Bien luu tru da capture background chua

# Camera
camera = cv2.VideoCapture(0)
camera.set(10, 200)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01)

dem = 1
while camera.isOpened():
    # Doc anh tu webcam
    ret, frame = camera.read()
    # Lam min anh
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # Lat ngang anh
    frame = cv2.flip(frame, 1)

    # Ve khung hinh chu nhat vung detection region
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    # Neu ca capture dc nen
    if isBgCaptured == 1:
        # Tach nen
        img = remove_background(frame)

        # Lay vung detection
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

        # Chuyen ve den trang
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshRe = cv2.resize(thresh, dsize=None, fx=0.5, fy=0.5)
        cv2.imshow('thresh', threshRe)  # anh da remove background

        # Tên folder của ký tự cần lấy ảnh train
        foldr = 'Z'

        filename = '../MiAI_Hand_Lang/' + foldr + '/' + foldr + '_%03d.jpg' % (dem)
        dem += 1
        if dem > 600:
            break
        cv2.imwrite(filename, threshRe)

    # Xu ly phim bam
    k = cv2.waitKey(100)

    if k == ord('q'):  # Bam q de thoat
        break
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

        isBgCaptured = 1
        cv2.putText(frame, "Background captured", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 255), 10, lineType=cv2.LINE_AA)
        time.sleep(2)
        print('Background captured')
    elif k == ord('z'):
        dem = 1
    elif k == ord('r'):
        bgModel = None
        isBgCaptured = 0
        cv2.putText(frame, "Background reset", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                    (0, 0, 255), 10, lineType=cv2.LINE_AA)
        print('Background reset')
        time.sleep(1)

    cv2.imshow('original', cv2.resize(frame, dsize=None, fx=0.5, fy=0.5))

cv2.destroyAllWindows()
camera.release()
