'''
Chạy file này với lệnh python <file name.py> để dùng thêm các tuỳ chọn parse
Vd: Muốn lưu video record với tên khác Output.avi:
>> python <file name.py> --name <tên mới>
'''

import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import time
import datetime as dt
import argparse
import numpy as np
from keras.models import load_model

# Load model tu file da train
model1 = load_model('models/mymodel_rgb_1v2.h5')
model2 = load_model('models/mymodel_rgb_2v2.h5')
model3 = load_model('models/mymodel_rgb_3v4.h5')
model4 = load_model('models/mymodel_rgb_4v2.h5')

# Cac khai bao bien
prediction = ''
score = 0

gesture_names1 = {0: 'A',
                  1: 'B',
                  2: 'C',
                  3: 'D',
                  4: 'E',
                  5: 'F'}

gesture_names2 = {0: 'G',
                  1: 'H',
                  2: 'I',
                  3: 'J',
                  4: 'K',
                  5: 'L',
                  6: 'M'}

gesture_names3 = {0: 'N',
                  1: 'O',
                  2: 'P',
                  3: 'Q',
                  4: 'R',
                  5: 'S'}

gesture_names4 = {0: 'T',
                  1: 'U',
                  2: 'V',
                  3: 'W',
                  4: 'X',
                  5: 'Y',
                  6: 'Z'}

# Khai bao kich thuoc vung detection region
cap_region_x_begin = 0.5
cap_region_y_end = 0.8

# Cac thong so lay threshold
threshold = 60
blurValue = 41
bgSubThreshold = 50  # 50
learningRate = 0

# Nguong du doan ky tu
predThreshold = 50


# isBgCaptured = 0  # Bien luu tru da capture background chua


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.ok = False
        self.isBgCaptured = 0
        self.gesture_names = gesture_names1
        self.model = model1
        self.bgModel = None

        # timer
        self.timer = ElapsedTimeClock(self.window)

        # open video source (by default this will try to open the computer webcam)
        self.vid = VideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tk.Button(window, text="Snapshot", fg='red', command=self.snapshot)
        self.btn_snapshot.pack(side=tk.LEFT)

        # video control buttons

        self.btn_start = tk.Button(window, text='START RECORD', bg='#00FFDD', command=self.open_camera)
        self.btn_start.pack(side=tk.LEFT)

        self.btn_stop = tk.Button(window, text='STOP RECORD', bg='#00FFDD', command=self.close_camera)
        self.btn_stop.pack(side=tk.LEFT)

        # remove BG button
        self.btn_removeBG = tk.Button(window, text='Remove BG & Start Detect', bg='#E8B252', command=self.removeBG)
        self.btn_removeBG.pack(side=tk.LEFT)

        # stop detect button
        self.btn_stopDetect = tk.Button(window, text='Stop Detect', bg='#E8B252', command=self.stopDetect)
        self.btn_stopDetect.pack(side=tk.LEFT)

        # Nhận diện A - F
        self.btn_model1 = tk.Button(window, text='A - F', command=self.btn_model1)
        self.btn_model1.pack(side=tk.LEFT)

        # Nhận diện G - M
        self.btn_model2 = tk.Button(window, text='G - M', command=self.btn_model2)
        self.btn_model2.pack(side=tk.LEFT)

        # Nhận diện N - S
        self.btn_model3 = tk.Button(window, text='N - S', command=self.btn_model3)
        self.btn_model3.pack(side=tk.LEFT)

        # Nhận diện T - Z
        self.btn_model4 = tk.Button(window, text='T - Z', command=self.btn_model4)
        self.btn_model4.pack(side=tk.LEFT)

        # quit button
        self.btn_quit = tk.Button(window, text='QUIT', bg='gray', fg='white', command=quit)
        self.btn_quit.pack(side=tk.LEFT)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 10
        self.update()

        self.window.mainloop()

    # Ham de predict xem la ky tu gi
    def predict_rgb_image_vgg(self, image):
        image = np.array(image, dtype='float32')
        image /= 255
        pred_array = self.model.predict(image)
        print(f'pred_array: {pred_array}')
        result = self.gesture_names[np.argmax(pred_array)]
        print(f'Result: {result}')
        print(max(pred_array[0]))
        score = float("%0.2f" % (max(pred_array[0]) * 100))
        print(result)
        return result, score

    # Ham xoa nen khoi anh
    def remove_background(self, frame):
        fgmask = self.bgModel.apply(frame, learningRate=learningRate)
        kernel = np.ones((3, 3), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        return res

    def removeBG(self):
        self.isBgCaptured = 1
        self.bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        print('Background captured')

    def stopDetect(self):
        self.bgModel = None
        self.isBgCaptured = 0
        print('Detect stopped.')
        print('Background reset.')

    def btn_model1(self):
        self.model = model1
        self.gesture_names = gesture_names1
        print('Changed model 1.')

    def btn_model2(self):
        self.model = model2
        self.gesture_names = gesture_names2
        print('Changed model 2.')

    def btn_model3(self):
        self.model = model3
        self.gesture_names = gesture_names3
        print('Changed model 3.')

    def btn_model4(self):
        self.model = model4
        self.gesture_names = gesture_names4
        print('Changed model 4.')

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def open_camera(self):
        self.ok = True
        self.timer.start()
        print("camera opened => Recording")

    def close_camera(self):
        self.ok = False
        self.timer.stop()
        print("camera closed => Not Recording")

    def update(self):

        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        # Lam min anh
        frame = cv2.bilateralFilter(frame, 5, 50, 100)
        # Lat ngang anh
        frame = cv2.flip(frame, 1)

        # Ve khung hinh chu nhat vung detection region
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

        # Neu ca capture dc nen
        if self.isBgCaptured == 1:
            # Tach nen
            img = self.remove_background(frame)

            # Lay vung detection
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                  int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

            # Chuyen ve den trang
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

            # cv2.imshow('original1', cv2.resize(blur, dsize=None, fx=0.5, fy=0.5))

            _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # cv2.imshow('thresh', cv2.resize(thresh, dsize=None, fx=0.5, fy=0.5))

            if (np.count_nonzero(thresh) / (thresh.shape[0] * thresh.shape[0]) > 0.2):
                # Neu nhu ve duoc hinh ban tay
                if (thresh is not None):
                    # Dua vao mang de predict
                    target = np.stack((thresh,) * 3, axis=-1)
                    target = cv2.resize(target, (224, 224))
                    target = target.reshape(1, 224, 224, 3)
                    prediction, score = self.predict_rgb_image_vgg(target)

                    # Neu probality > nguong du doan thi hien thi
                    print(score, prediction)
                    if (score >= predThreshold):
                        cv2.putText(frame, "Sign:" + prediction, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 3,
                                    (0, 0, 255), 10, lineType=cv2.LINE_AA)
        thresh = None
        if self.ok:
            self.vid.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)


class VideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Command Line Parser
        args = CommandLineParser().args

        # create videowriter

        # 1. Video Type
        VIDEO_TYPE = {
            'avi': cv2.VideoWriter_fourcc(*'XVID'),
            # 'mp4': cv2.VideoWriter_fourcc(*'H264'),
            'mp4': cv2.VideoWriter_fourcc(*'XVID'),
        }

        self.fourcc = VIDEO_TYPE[args.type[0]]

        # 2. Video Dimension
        STD_DIMENSIONS = {
            '480p': (640, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080),
            '4k': (3840, 2160),
        }
        res = STD_DIMENSIONS[args.res[0]]
        print(args.name, self.fourcc, res)
        self.out = cv2.VideoWriter(args.name[0] + '.' + args.type[0], self.fourcc, 10, res)

        # set video sourec width and height
        self.vid.set(3, res[0])
        self.vid.set(4, res[1])

        # Get video source width and height
        self.width, self.height = res

    # To get frames
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                return (ret, None)
        else:
            return (None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            self.out.release()
            cv2.destroyAllWindows()


class ElapsedTimeClock:
    def __init__(self, window):
        self.T = tk.Label(window, text='00:00:00', font=('times', 20, 'bold'), bg='green')
        # self.T.pack(fill=tk.BOTH, expand=1)
        self.elapsedTime = dt.datetime(1, 1, 1)
        self.running = 0
        self.lastTime = ''
        t = time.localtime()
        self.zeroTime = dt.timedelta(hours=t[3], minutes=t[4], seconds=t[5])
        # self.tick()

    def tick(self):
        # get the current local time from the PC
        self.now = dt.datetime(1, 1, 1).now()
        self.elapsedTime = self.now - self.zeroTime
        self.time2 = self.elapsedTime.strftime('%H:%M:%S')
        # if time string has changed, update it
        if self.time2 != self.lastTime:
            self.lastTime = self.time2
            self.T.config(text=self.time2)
        # calls itself every 200 milliseconds
        # to update the time display as needed
        # could use >200 ms, but display gets jerky
        self.updwin = self.T.after(100, self.tick)

    def start(self):
        if not self.running:
            self.zeroTime = dt.datetime(1, 1, 1).now() - self.elapsedTime
            self.tick()
            self.running = 1

    def stop(self):
        if self.running:
            self.T.after_cancel(self.updwin)
            self.elapsedTime = dt.datetime(1, 1, 1).now() - self.zeroTime
            self.time2 = self.elapsedTime
            self.running = 0


# Tuỳ chọn thêm

class CommandLineParser:

    def __init__(self):
        # Create object of the Argument Parser
        parser = argparse.ArgumentParser(description='Script to record videos')

        # Create a group for requirement
        # for now no required arguments
        # required_arguments=parser.add_argument_group('Required command line arguments')

        # Only values is supporting for the tag --type. So nargs will be '1' to get
        parser.add_argument('--type', nargs=1, default=['avi'], type=str,
                            help='Type of the video output: for now we have only AVI & MP4')

        # Only one values are going to accept for the tag --res. So nargs will be '1'
        parser.add_argument('--res', nargs=1, default=['480p'], type=str,
                            help='Resolution of the video output: for now we have 480p, 720p, 1080p & 4k')

        # Only one values are going to accept for the tag --name. So nargs will be '1'
        parser.add_argument('--name', nargs=1, default=['output'], type=str, help='Enter Output video title/name')

        # Parse the arguments and get all the values in the form of namespace.
        # Here args is of namespace and values will be accessed through tag names
        self.args = parser.parse_args()


def main():
    # Create a window and pass it to the Application object
    App(tk.Tk(), 'Video Recorder')


main()
