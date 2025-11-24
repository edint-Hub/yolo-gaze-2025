from eye_tracking_valid.mp_iris_area import MPIrisArea


def demo_iris_area():
    mp_iris_area = MPIrisArea()
    mp_iris_area.tracking_streaming(webcam_port=0, visible=True)


if __name__ == "__main__":
    demo_iris_area()
