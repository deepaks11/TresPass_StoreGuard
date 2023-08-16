import os
from typing import Dict
import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO
from multiprocessing.pool import ThreadPool
from main import YoloObjectDetection
from intersection import TressPass
queue_name = queue.Queue()
pool = ThreadPool(processes=1)
root = os.getcwd()


class VideoCapture:

    def __init__(self, rtsp_url):

        self.cap = cv2.VideoCapture(rtsp_url)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True  # Set the thread as daemon to stop it when the main program exits
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()


class PlayVideo:

    def __init__(self, source, window_name, q):
        self.cap_line = None
        self.model = YOLO("yolov8s.pt")
        self.detections = None
        self.image = None
        self.obj = None
        self.frame = None
        self.resized_img = None
        self.source = source
        self.window_name = window_name
        self.line_coordinates = (327, 60), (349, 200)
        self.tracker_state: Dict[str, bool] = {}
        self.q_img = q
        self.points = [(0, 0), (0, 480), (395, 480), (333, 0)]

    def vdo_cap(self):

        try:

            if self.source.startswith("rtsp"):
                self.cap_line = VideoCapture(self.source)
            else:
                self.cap_line = cv2.VideoCapture(self.source)

            while True:
                # if you want to read any different video file format just add below
                if self.source.endswith((".mp4", ".avi")):
                    ret, self.image = self.cap_line.read()

                else:
                    self.image = self.cap_line.read()

                self.image = cv2.resize(self.image, (640, 480))
                # height, width, _ = self.image.shape
                # crop_img = self.image[:, width//2:]
                mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(self.points)], 255)
                crop_frame = cv2.bitwise_and(self.image, self.image, mask=mask)
                x, y, w, h = cv2.boundingRect(np.array(self.points))

                # Crop the result to the bounding rectangle
                cropped_img = crop_frame[y:y + h, x:x + w]

                self.q_img.put(cropped_img)

                # passing frame to object detection and tracking
                obj = pool.apply_async(YoloObjectDetection(self.q_img, self.model).predict)
                frame, detections = obj.get()
                check_intersect = TressPass(detections, self.line_coordinates).intersection()
                result = self.image.copy()
                result[np.where(mask > 0)] = frame[np.where(mask > 0)]
                cv2.line(img=result, pt1=(327, 60), pt2=(349, 200), color=(255, 0, 0), thickness=2)
                # self.image[:, width // 2:] = frame
                if check_intersect:
                    cv2.putText(img=result, text="Tress_Pass_detected", org=(35, 35),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=4)
                else:
                    pass
                cv2.imshow(self.window_name, result)
                if cv2.waitKey(1) == ord('q'):  # Exit if 'q' is pressed
                    break
        except Exception as e:
            print(e)


if __name__ == "__main__":
    urls = [

        # {"name": "rtsp1", "url": r"filename1.avi"},
        {"name": "rtsp2", "url": r"2023-07-28_16-00-17.mp4"},
        # {"name": "rtsp3", "url": r"rtsp://admin:Admin123$@10.11.25.64:554/stream1"},
        # {"name": "rtsp4", "url": r"rtsp://admin:Admin123$@10.11.25.57:554/stream1"},
        # {"name": "rtsp5", "url": r"rtsp://admin:Admin123$@10.11.25.59:554/stream1"},
        # {"name": "rtsp6", "url": r"rtsp://admin:Admin123$@10.11.25.51:554/stream1"},
        # {"name": "rtsp7", "url": r"rtsp://admin:Admin123$@10.11.25.52:554/stream1"},
        # {"name": "rtsp8", "url": r"rtsp://admin:Admin123$@10.11.25.53:554/stream1"},
        # {"name": "rtsp9", "url": r"rtsp://admin:Admin123$@10.11.25.54:554/stream1"},
        # {"name": "rtsp10", "url": r"rtsp://admin:Admin123$@10.11.25.55:554/stream1"},
        # {"name": "rtsp11", "url": r"rtsp://admin:Admin123$@10.11.25.59:554/stream1"},
        # {"name": "rtsp12", "url": r"rtsp://admin:Admin123$@10.11.25.63:554/stream1"}
    ]
    queue_list = []
    threads = []
    for i in urls:
        url = i['url']
        name = i["name"]
        queue_name.name = name
        queue_list.append(queue_name)
        td = threading.Thread(
            target=PlayVideo(url, name, queue_name).vdo_cap)
        td.start()
        threads.append(td)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()