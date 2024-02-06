import sys
import time
import torch
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QCoreApplication, QTimer
import cv2



# UI파일 연결
form_class = uic.loadUiType("main.ui")[0]


# 화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Yolov5Interface')
        self.resize(1000, 650)

        # 화면 중앙으로
        qr = self.frameGeometry()
        qr.moveCenter(QDesktopWidget().availableGeometry().center())
        self.move(qr.topLeft())
        self.show()

        self.weights = ('./best.pt')   # 기본 가중치
        self.model = self.loadModel()  # Yolo model 가져오기
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'   # cuda or cpu
        self.classes = self.model.names
        self.video = 0
        # self.url = 'https://www.youtube.com/watch?v=5YymvwfhPm0'
        # self.video = pafy.new(self.url)
        # self.b = self.video.getbest(preftype='mp4')

        # combobox video type
        self.cmbVideoType.activated[str].connect(self.changeVideoType)
        self.btnOpen.setDisabled(True)

        # create a timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)  # 스레드 처럼 사용


        self.btnStart.clicked.connect(self.startTime)

        # menu bar button
        self.menu_setting_exit.triggered.connect(self.menuSettingExitFunction)
        self.menu_help_info.triggered.connect(self.menuHelpInfoFunction)


    def changeVideoType(self, text):
        if text == 'cam':
            self.btnOpen.setDisabled(True)
        elif text =='video':
            self.btnOpen.setEnabled(True)

    def startTime(self):
        # if timer is stopped
        if not self.timer.isActive():
            # create video capture
            self.cap = cv2.VideoCapture(self.video)
            self.cap.set(cv2.CAP_PROP_FPS, 120)
            # start timer
            self.timer.start(20)


    def viewCam(self):
        start = time.time()

        ret, img = self.cap.read()
        results = self.scoreFrame(img)
        img = self.plotBoxes(results, img)

        height, width, channel = img.shape
        step = channel * width
        # QImage 생성
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qImg = QImage(img.data, width, height, step, QImage.Format_RGB888)
        # 최종 이미지 출력
        self.labelMain.setPixmap(QPixmap.fromImage(qImg))

        # 프레임 표시
        stop = time.time()
        interval = stop - start
        fps = str(int((1 / interval)))
        self.labelFps.setText(f'FPS : {fps}')

    # 객체 탐지
    def scoreFrame(self, img):
        self.model.to(self.device)
        results = self.model([img])
        labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
        return labels, cord
    
    # 클래스 라벨 반환
    def classToLabel(self, i):
        return self.classes[int(i)]
    
    # 결과 화면에 그리기
    def plotBoxes(self, results, img):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = img.shape[1], img.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(img, self.classToLabel(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return img
    
    # 모델 가져오기
    def loadModel(self):
        model = torch.hub.load('.', 'custom', path=self.weights, source='local')
        return model

    # 프로그램 종료
    def menuSettingExitFunction(self):
        QCoreApplication.instance().quit()

    # 프로그램 정보창
    def menuHelpInfoFunction(self):
        print('info')



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()