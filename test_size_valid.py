from model_size_vaild.model_valid_cli import YOLOPredictValidCLI
import argparse


def test_valid_cli(args):
    yolo_predict_cli = YOLOPredictValidCLI()
    yolo_predict_cli.run(args.origin_model_path, args.convert_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Model Time Valid Test")

    parser.add_argument(
        "--origin_model_path", required=True, help="Path to the origin model"
    )
    parser.add_argument(
        "--convert_model_path", required=True, help="Path to the convert model"
    )

    args = parser.parse_args()
    test_valid_cli(args)
