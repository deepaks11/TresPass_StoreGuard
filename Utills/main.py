import supervision as sv
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class YoloObjectDetection:

    def __init__(self, q_img, model):

        self.q_img = q_img.get()
        self.model = model
        # self.model = model

    def predict(self):
        try:

            box_annotator = sv.BoxAnnotator(
                thickness=2,
                text_thickness=1,
                text_scale=0.5
            )

            for result in self.model(source=self.q_img, agnostic_nms=True, classes=0, verbose=False):

                frame = result.orig_img
                detections = sv.Detections.from_yolov8(result)

                if result.boxes.id is not None:
                    detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

                labels = [
                    f"{self.model.names[class_id]}"
                    for _, confidence, class_id, tracker_id
                    in detections
                ]

                frame = box_annotator.annotate(
                    scene=frame,
                    detections=detections,
                    labels=labels
                )
                return frame, detections

        except Exception as er:
            print(er)
