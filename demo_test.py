from eye_tracking_valid.mp_iris_area import MPIrisArea
from model_size_vaild.model_vaild_gui import YOLOPredictorGUI
from analysis_time_valid.time_valid_gui import YOLOPredictTimeValidGUI


def demo_iris_area():
    mp_iris_area = MPIrisArea()
    mp_iris_area.tracking_streaming(webcam_port=0, visible=True)


def demo_model_size():
    yolo_predict_gui = YOLOPredictorGUI()
    yolo_predict_gui.mainloop()


def demo_model_time():
    yolo_predict_time_gui = YOLOPredictTimeValidGUI()
    yolo_predict_time_gui.mainloop()


if __name__ == "__main__":
    demo_iris_area()
    # demo_model_size()
    # demo_model_time()
