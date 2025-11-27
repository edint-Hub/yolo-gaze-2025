from ultralytics import YOLO

import os
import cv2
import time
import torch


class YOLOPredictTimeCLI:
    origin_model_path = None
    convert_model_path = None

    origin_model: YOLO | None = None
    convert_model: YOLO | None = None

    video_path = os.path.join("datasets", "video", "test.mp4")

    def __init__(self):
        pass

    def run(self, origin_model_path: str, convert_model_path: str, device: str = "cpu"):
        origin_model = YOLO(origin_model_path)
        convert_model = YOLO(convert_model_path)

        x = torch.rand(1, 3, 640, 640).to(device)

        print(f"AI WarmUp Start..")
        origin_model.predict(x, verbose=False)
        convert_model.predict(x, verbose=False)

        if os.path.exists(self.video_path):
            print("Start to predict time..")
            print("Origin Model Predict Start.")
            origin_time = self.__predict(origin_model, visible=True)
            print("Convert Model Predict Start.")
            convert_time = self.__predict(convert_model, visible=True)
            print("==" * 20)
            print(f"Origin Model Time : {origin_time} sec")
            print(f"Convert Model Time : {convert_time} sec")
            print(
                f"Origin - Convert Model Time Diff : {int(origin_time - convert_time)} sec"
            )
            print(
                f"Origin - Convert Model Time Diff Percent : {(origin_time - convert_time) / origin_time * 100:.2f}%"
            )
            print()
        else:
            print(f"Can't found video file. Check this path : {self.video_path}")

    def __predict(self, model: YOLO, visible: bool = False):
        cap = cv2.VideoCapture(self.video_path)
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            predict_frame = model.predict(frame, verbose=False)[0].plot()
            if visible:
                cv2.imshow("Predict Video", predict_frame)
                cv2.waitKey(1)

        end_time = time.time()
        cap.release()
        if visible:
            cv2.destroyAllWindows()
        return int(end_time - start_time)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="AI Model Time Valid Test")

#     parser.add_argument(
#         "--origin_model_path", required=True, help="Path to the origin model"
#     )
#     parser.add_argument(
#         "--convert_model_path", required=True, help="Path to the convert model"
#     )

#     args = parser.parse_args()

#     yolo_predict_cli = YOLOPredictTimeCLI()
#     yolo_predict_cli.run(args.origin_model_path, args.convert_model_path)
