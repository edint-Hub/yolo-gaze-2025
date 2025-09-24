import cv2
from ultralytics import YOLO


def main():
    # 기본 YOLO 모델 로드 (객체 검출만)
    model = YOLO("yolo11n.pt")

    # 웹캠 캡처 시작
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    print("실시간 객체 검출을 시작합니다. 'q' 키를 눌러 종료하세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 좌우 반전
        frame = cv2.flip(frame, 1)

        # YOLO 추론
        results = model.predict(source=frame, conf=0.5, verbose=False)

        # 결과 시각화
        annotated_frame = results[0].plot()

        # 검출된 객체 수 표시
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated_frame, f"Objects: {detections}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 프레임 표시
        cv2.imshow("Simple Object Detection", annotated_frame)

        # 'q' 키로 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()