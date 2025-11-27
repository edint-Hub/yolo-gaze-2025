import cv2
import mediapipe as mp
import numpy as np
import math
import os
import glob


class MPIrisArea:
    SENSITIVITY = 0.8
    BASE_SCALE = 50

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.calib_offset = np.array([0.0, 0.0])
        self.calibrated = False

        self.avg_vector = None
        self.gaze_point = None
        self.avg_eye_center = None
        self.arrow_tip = None
        self.calibrated_angle = None

        self.cell_x = None
        self.cell_y = None
        self.cell_name = None

    def get_2d_point(self, landmark, width: int, height: int):
        return np.array([int(landmark.x * width), int(landmark.y * height)])

    def valid_with_image(self, img_folder_path: str, visible_save: bool = False):
        if not os.path.exists(img_folder_path):
            print(f"Folder Not Exist : {img_folder_path}")
            return

        valid_result_list = []
        valid_success_count = 0
        valid_failed_count = 0
        valid_error_count = 0

        pattern = os.path.join(img_folder_path, "calibration_*.jpg")
        files = glob.glob(pattern)
        origin_calib_offset = None

        if files:
            x_offset = float(files[0].split("_")[1])
            y_offset = float(files[0].split("_")[2].split(".jpg")[0])
            self.calib_offset = np.array([x_offset, y_offset])
            origin_calib_offset = self.calib_offset.copy()
            print(f"Calibration OffSet : {self.calib_offset}")
            self.calibrated = True
        else:
            print("WARNING) Calibration Data is not apply!!!!")

        img_namd_list = os.listdir(img_folder_path)
        if len(img_namd_list) == 0:
            print("No Labeled Data!")
            return

        for img_name in img_namd_list:  # file name type : {idx}_{x}_{y}.jpg(png)
            self.__clear()
            if origin_calib_offset is not None:
                self.calib_offset = origin_calib_offset.copy()
                self.calibrated = True

            if img_name.startswith("calibration") or img_name.startswith("tracking"):
                continue

            img_path = os.path.join(img_folder_path, img_name)
            try:
                drawing_frame = self.tracking_image(img_path, visible=visible_save)
                if visible_save:
                    cv2.imwrite(
                        os.path.join(img_folder_path, f"tracking_{img_name}"),
                        drawing_frame,
                    )
                file_info = img_name.split("_")
                if (
                    int(file_info[1]) == self.cell_x
                    and int(file_info[2].split(".")[0]) == self.cell_y
                ):
                    valid_result_list.append({int(file_info[0]): True})
                    valid_success_count += 1
                    print("==" * 20)
                    print(
                        f'[Answer] x : {int(file_info[1])}, y : {int(file_info[2].split(".")[0])}'
                    )
                    print(f"[Predict] x : {self.cell_x}, y : {self.cell_y}")
                    print("[Result] : True")
                    print("==" * 20)
                    print()
                else:
                    valid_result_list.append({int(file_info[0]): False})
                    valid_failed_count += 1
                    print("==" * 20)
                    print(
                        f'[Answer] x : {int(file_info[1])}, y : {int(file_info[2].split(".")[0])}'
                    )
                    print(f"[Predict] x : {self.cell_x}, y : {self.cell_y}")
                    print("[Result] : False")
                    print("==" * 20)
                    print()
            except:
                valid_result_list.append({int(file_info[0]): False})
                valid_error_count += 1

        # print(f"valid result : {valid_result_list}")
        print(f"valid success count : {valid_success_count}")
        print(f"valid failed count : {valid_failed_count}")
        print(f"valid error count : {valid_error_count}")
        print(
            f"valid success percent : {valid_success_count / (valid_success_count + valid_failed_count + valid_error_count) * 100:.2f}"
        )
        return valid_result_list

    def tracking_image(self, img_path: str, visible: bool = True):
        if not os.path.exists(img_path):
            print(f"File Not Exist : {img_path}")
            return

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            self.__tracking_eye_sight(results, w, h)
            self.cell_x, self.cell_y, self.cell_name = self.__get_gaze_cell(w, h)
            if visible:
                drawing_frame = self.__drawing(
                    img,
                    self.avg_eye_center,
                    self.arrow_tip,
                    self.calibrated_angle,
                    self.gaze_point,
                    w,
                    h,
                )
            else:
                drawing_frame = img
        else:
            if visible:
                cv2.putText(
                    img,
                    "Face not detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                drawing_frame = img
            else:
                drawing_frame = img
        if visible:
            cv2.imshow("Calibrated Gaze Visualization", drawing_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return drawing_frame

    def tracking_video(self, video_path: str, visible: bool = True):
        if not os.path.exists(video_path):
            print(f"File Not Exist : {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                self.__tracking_eye_sight(results, w, h)

                if visible:
                    drawing_frame = self.__drawing(
                        frame,
                        self.avg_eye_center,
                        self.arrow_tip,
                        self.calibrated_angle,
                        self.gaze_point,
                        w,
                        h,
                    )
            else:
                if visible:
                    cv2.putText(
                        frame,
                        "Face not detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    drawing_frame = frame

            if visible:
                cv2.imshow("Calibrated Gaze Visualization", drawing_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            elif key == ord("c"):
                self.calib_offset = self.avg_vector.copy()
                self.calibrated = True
                print("Calibration set:", self.calib_offset)
            elif key == ord("r"):
                self.calibrated = False
                self.calib_offset = np.array([0.0, 0.0])
                print("Calibration reset.")

        cap.release()
        if visible:
            cv2.destroyAllWindows()

    def tracking_streaming(self, webcam_port: int, visible: bool = True):
        cap = cv2.VideoCapture(webcam_port)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]

            origin_frame = frame.copy()
            total_count += 1

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            if results.multi_face_landmarks:

                self.__tracking_eye_sight(results, w, h)
                self.cell_x, self.cell_y, self.cell_name = self.__get_gaze_cell(w, h)

                if visible:
                    drawing_frame = self.__drawing(
                        frame,
                        self.avg_eye_center,
                        self.arrow_tip,
                        self.calibrated_angle,
                        self.gaze_point,
                        w,
                        h,
                    )

            else:
                if visible:
                    cv2.putText(
                        frame,
                        "Face not detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    drawing_frame = frame

            if visible:
                cv2.imshow("Calibrated Gaze Visualization", drawing_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            elif key == ord("c"):
                self.calib_offset = self.avg_vector.copy()
                self.calibrated = True
            elif key == ord("r"):
                self.calibrated = False
                self.calib_offset = np.array([0.0, 0.0])
                print("Calibration reset.")

        cap.release()
        if visible:
            cv2.destroyAllWindows()

    def make_label_data_streaming(
        self,
        webcam_port: int,
        visible: bool = True,
        save_path: str | None = None,
    ):
        cap = cv2.VideoCapture(webcam_port)
        fps = cap.get(cv2.CAP_PROP_FPS)

        total_count = 0
        save_count = int(fps / 3)
        img_idx = 0
        skip_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]

            origin_frame = frame.copy()
            total_count += 1

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                if skip_count % 3 == 0:
                    self.__tracking_eye_sight(results, w, h)
                    self.cell_x, self.cell_y, self.cell_name = self.__get_gaze_cell(
                        w, h
                    )
                    skip_count = 0
                else:
                    skip_count += 1

                if visible:
                    drawing_frame = self.__drawing(
                        frame,
                        self.avg_eye_center,
                        self.arrow_tip,
                        self.calibrated_angle,
                        self.gaze_point,
                        w,
                        h,
                        img_idx,
                    )

                if (
                    self.calibrated
                    and total_count % save_count == 0
                    and save_path is not None
                    and self.cell_x is not None
                    and self.cell_y is not None
                ):
                    cv2.imwrite(
                        os.path.join(
                            save_path, f"{img_idx}_{self.cell_x}_{self.cell_y}.jpg"
                        ),
                        origin_frame,
                    )
                    img_idx += 1
                    print(f"[{img_idx}] Image saved.")
            else:
                if visible:
                    cv2.putText(
                        frame,
                        "Face not detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    drawing_frame = frame

            if visible:
                cv2.imshow("Calibrated Gaze Visualization", drawing_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            elif key == ord("c"):
                self.calib_offset = self.avg_vector.copy()
                self.calibrated = True
                if save_path is not None:
                    cv2.imwrite(
                        os.path.join(
                            save_path,
                            f"calibration_{self.calib_offset[0]}_{self.calib_offset[1]}.jpg",
                        ),
                        origin_frame,
                    )
                    print("Calibration set saved.")
                print("Calibration set:", self.calib_offset)
            elif key == ord("s"):
                if (
                    save_path is not None
                    and self.cell_x is not None
                    and self.cell_y is not None
                ):
                    cv2.imwrite(
                        os.path.join(
                            save_path, f"{save_count}_{self.cell_x}_{self.cell_y}.jpg"
                        ),
                        origin_frame,
                    )
                    save_count += 1
                    print("Image saved.")
            elif key == ord("r"):
                self.calibrated = False
                self.calib_offset = np.array([0.0, 0.0])
                print("Calibration reset.")

        cap.release()
        if visible:
            cv2.destroyAllWindows()

    def __tracking_eye_sight(self, mp_results, width: int, height: int):
        face_landmarks = mp_results.multi_face_landmarks[0].landmark

        left_inner = self.get_2d_point(face_landmarks[133], width, height)
        left_outer = self.get_2d_point(face_landmarks[33], width, height)
        left_eye_center = (left_inner + left_outer) / 2

        left_iris_points = np.array(
            [
                self.get_2d_point(face_landmarks[idx], width, height)
                for idx in [474, 475, 476, 477]
            ]
        )
        left_iris_center = np.mean(left_iris_points, axis=0)
        left_vector = left_iris_center - left_eye_center

        right_inner = self.get_2d_point(face_landmarks[362], width, height)
        right_outer = self.get_2d_point(face_landmarks[263], width, height)
        right_eye_center = (right_inner + right_outer) / 2

        right_iris_points = np.array(
            [
                self.get_2d_point(face_landmarks[idx], width, height)
                for idx in [469, 470, 471, 472]
            ]
        )
        right_iris_center = np.mean(right_iris_points, axis=0)
        right_vector = right_iris_center - right_eye_center

        self.avg_eye_center = (left_eye_center + right_eye_center) / 2
        self.avg_vector = (left_vector + right_vector) / 2

        if self.calibrated:
            calibrated_vector = self.avg_vector - self.calib_offset
        else:
            calibrated_vector = self.avg_vector

        self.calibrated_angle = math.degrees(
            math.atan2(calibrated_vector[1], calibrated_vector[0])
        )

        # 시선 벡터 따라 도달할 지점 (가상 '화면에서 바라보는 지점')
        self.arrow_tip = (
            self.avg_eye_center + calibrated_vector * self.BASE_SCALE * self.SENSITIVITY
        ).astype(int)

        self.gaze_point = self.arrow_tip  # 여기서 gaze point를 얻음

    def __get_gaze_cell(self, width, height):
        if self.gaze_point is None:
            return None, None, "Unknown"

        # 안전한 좌표 보정
        gx = np.clip(self.gaze_point[0], 0, width - 1)
        gy = np.clip(self.gaze_point[1], 0, height - 1)

        # 각 셀 크기
        cell_w = width // 3
        cell_h = height // 3

        # 셀 인덱스 계산 (0~2)
        cell_x = gx // cell_w  # 0: Left,   1: Center, 2: Right
        cell_y = gy // cell_h  # 0: Top,    1: Middle, 2: Bottom
        try:
            # 좌표 기반 이름 매핑
            name_map_x = ["Left", "Center", "Right"]
            name_map_y = ["Top", "Middle", "Bottom"]

            cell_name = f"{name_map_y[cell_y]}-{name_map_x[cell_x]}"

            return int(cell_x), int(cell_y), cell_name
        except:
            return None, None, "Unknown"

    def __drawing(
        self,
        frame: np.ndarray,
        avg_eye_center,
        arrow_tip,
        calibrated_angle,
        gaze_point,
        width: int,
        height: int,
        idx: int = -1,
    ):
        # ----------------- 3x3 그리드 영역 처리 -----------------
        cell_w = width // 3
        cell_h = height // 3
        # gaze_point가 None이 아니면, 해당 셀 인덱스 구하기
        if gaze_point is not None:
            gx = np.clip(gaze_point[0], 0, width - 1)
            gy = np.clip(gaze_point[1], 0, height - 1)
            cell_x = gx // cell_w
            cell_y = gy // cell_h
        else:
            cell_x = cell_y = None

        # 반투명 오버레이 준비
        overlay = frame.copy()
        alpha = 0.4

        # 3x3 그리드 그리기 + gaze 셀 색칠
        for i in range(3):
            for j in range(3):
                x1 = i * cell_w
                y1 = j * cell_h
                x2 = (i + 1) * cell_w
                y2 = (j + 1) * cell_h

                # gaze가 이 셀 안에 있다면 반투명 색칠
                if cell_x == i and cell_y == j:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)

                # 셀 경계선 그리기 (흰색)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # 반투명 오버레이 합성
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # 시각화 (화살표, 원, 각도 등)
        cell_x, cell_y, cell_name = self.__get_gaze_cell(width, height)

        cv2.arrowedLine(
            frame,
            tuple(avg_eye_center.astype(int)),
            tuple(arrow_tip),
            (255, 0, 0),
            2,
        )
        cv2.circle(frame, tuple(avg_eye_center.astype(int)), 3, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"Avg Angle: {calibrated_angle:.1f}, Gaze Cell: ({cell_x}, {cell_y}) -> {cell_name}",
            tuple(avg_eye_center.astype(int) + np.array([0, -10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cal_text = "Calibrated" if self.calibrated else "Not Calibrated"
        idx_str = "" if idx == -1 else f"idx : [{idx}]"
        cv2.putText(
            frame,
            f"[{cal_text}] Press 'c' to calibrate, 'r' to reset. {idx_str}",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        return frame

    def __clear(self):
        self.calib_offset = np.array([0.0, 0.0])
        self.calibrated = False

        self.avg_vector = None
        self.gaze_point = None
        self.avg_eye_center = None
        self.arrow_tip = None
        self.calibrated_angle = None

        self.cell_x = None
        self.cell_y = None
        self.cell_name = None
