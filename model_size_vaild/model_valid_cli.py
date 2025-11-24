from typing import Literal
from ultralytics import YOLO

import os
import cv2
import torch


class YOLOPredictValidCLI:
    origin_model_path = None
    convert_model_path = None

    origin_model: YOLO | None = None
    convert_model: YOLO | None = None

    img_path = os.path.join("datasets", "image", "test.jpg")
    default_path = os.path.join("datasets", "image")

    def __init__(self):
        pass

    def run(self, origin_model_path: str, convert_model_path: str, device: str = "cpu"):
        origin_model = YOLO(origin_model_path)
        convert_model = YOLO(convert_model_path)

        x = torch.randn(1, 3, 640, 640).to(device)
        print(f"AI WarmUp Start..")
        origin_model.predict(x, verbose=False)
        convert_model.predict(x, verbose=False)

        if os.path.exists(self.img_path):
            self.__predict(origin_model, "origin")
            self.__predict(convert_model, "convert")
            print("==" * 20)
            print("Predict Finished.")
            print()
        else:
            print(f"Can't found image file. Check this path : {self.img_path}")

    def __predict(
        self,
        model: YOLO,
        model_type: Literal["origin", "convert"],
        is_img_save: bool = True,
    ):
        img = cv2.imread(self.img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        predict_result = model.predict(img_rgb, verbose=False, conf=0.4)[0]
        if is_img_save:
            predict_result.save(
                show=False,
                save=os.path.join(self.default_path, f"{model_type}_predict_img.jpg"),
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Model Valid Test")

    parser.add_argument(
        "--origin_model_path", required=True, help="Path to the origin model"
    )
    parser.add_argument(
        "--convert_model_path", required=True, help="Path to the convert model"
    )

    args = parser.parse_args()

    yolo_predict_cli = YOLOPredictValidCLI()
    yolo_predict_cli.run(args.origin_model_path, args.convert_model_path)
