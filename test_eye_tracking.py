from eye_tracking_valid.mp_iris_area import MPIrisArea

import os
import argparse


def test_eye_tracking_cli(args):
    img_folder_path = os.path.join("datasets", "image", "eye_tracking")
    mp_iris_tracking = MPIrisArea()
    mp_iris_tracking.valid_with_image(
        img_folder_path=img_folder_path, visible_save=args.visible_save
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eye Tracking Valid Test")

    parser.add_argument(
        "--visible_save",
        required=False,
        help="Tracking img save option",
        default=False,
        type=bool,
    )

    args = parser.parse_args()
    test_eye_tracking_cli(args=args)
