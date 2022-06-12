import cv2
import numpy as np
import time

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
# Giữ khoảng cách tay đến camera gần (tay chiếm gần hết khung cn)
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
        img = frame

        # Lay vung detection
        img = img[0 + 2:int(cap_region_y_end * frame.shape[0]) - 2,
              int(cap_region_x_begin * frame.shape[1]) + 2:frame.shape[1] - 2]  # clip the ROI

        # Chuyen ve den trang
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshRe = cv2.resize(thresh, dsize=None, fx=0.5, fy=0.5)
        cv2.imshow('original', img)  # anh da remove background

        # Tên folder của ký tự cần lấy ảnh train
        foldr = 'Z'

        filename = '../MiAI_Hand_Lang/' + foldr + '/' + foldr + '_%03d.jpg' % (dem)
        dem += 1
        if dem > 600:
            break
        # cv2.imwrite(filename, threshRe)
    # thresh = None

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
