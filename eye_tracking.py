import cv2
import mediapipe as mp
import numpy as np

# Mediapipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# 랜드마크와 이미지 크기를 기반으로 2D 좌표 계산 함수
def get_2d_point(landmark, width, height):
    return np.array([int(landmark.x * width), int(landmark.y * height)])


# 웹캠 캡처 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        # 왼쪽 눈과 오른쪽 눈의 중심점만 계산
        left_eye = get_2d_point(face_landmarks[33], w, h)  # 왼쪽 눈 외각
        right_eye = get_2d_point(face_landmarks[263], w, h)  # 오른쪽 눈 외각

        # 두 눈의 중심점
        eye_center = (left_eye + right_eye) // 2

        # 간단한 3x3 그리드
        cell_w = w // 3
        cell_h = h // 3

        # 눈 중심 위치를 기반으로 그리드 셀 결정
        cell_x = min(eye_center[0] // cell_w, 2)
        cell_y = min(eye_center[1] // cell_h, 2)

        # 3x3 그리드 그리기
        for i in range(3):
            for j in range(3):
                x1, y1 = i * cell_w, j * cell_h
                x2, y2 = (i + 1) * cell_w, (j + 1) * cell_h

                # 현재 셀 하이라이트
                if cell_x == i and cell_y == j:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # 눈 중심점 표시
        cv2.circle(frame, tuple(eye_center), 5, (0, 255, 0), -1)

        # 현재 위치 텍스트 표시
        cv2.putText(frame, f"Position: ({cell_x}, {cell_y})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Face not detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    frame = cv2.flip(frame, 1)  # 좌우 반전
    cv2.imshow("Simple Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()