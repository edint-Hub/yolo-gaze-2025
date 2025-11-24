from ultralytics import YOLO
from tkinter import filedialog  # 파일/폴더 선택창

import os
import customtkinter as ctk
import cv2
import time
import threading


class YOLOPredictTimeValidGUI(ctk.CTk):
    origin_model_path = None
    convert_model_path = None

    origin_model: YOLO | None = None
    convert_model: YOLO | None = None

    video_path = os.path.join("datasets", "video", "test.mp4")

    def __init__(self):
        super().__init__()
        self.title("YOLO Time Valid")
        self.geometry("1280x950")
        self.resizable(False, False)

        # ---------------- 상단: 두 개의 정사각형 버튼 ----------------
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=20)

        # 왼쪽 (원본 모델) 버튼 프레임
        left_frame = ctk.CTkFrame(button_frame)
        left_frame.pack(side="left", padx=60, pady=10)

        self.origin_button = ctk.CTkButton(
            left_frame,
            text="원본\n모델",
            width=120,
            height=120,
            command=self.select_origin_model,
        )
        self.origin_button.pack(pady=10)

        self.origin_label = ctk.CTkLabel(
            left_frame, text="선택된 경로: 없음", width=300, justify="center"
        )
        self.origin_label.pack(pady=10)

        # 오른쪽 (변환 모델) 버튼 프레임
        right_frame = ctk.CTkFrame(button_frame)
        right_frame.pack(side="left", padx=60, pady=10)

        self.convert_button = ctk.CTkButton(
            right_frame,
            text="변환\n모델",
            width=120,
            height=120,
            command=self.select_convert_model,
        )
        self.convert_button.pack(pady=10)

        self.convert_label = ctk.CTkLabel(
            right_frame, text="선택된 경로: 없음", width=300, justify="center"
        )
        self.convert_label.pack(pady=10)

        # ---------------- 중단: 결과/프로그레스 표시 영역 ----------------
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.pack(fill="both", expand=True, pady=10)

        # 왼쪽 결과 (origin model)
        self.origin_result_frame = ctk.CTkFrame(self.result_frame)
        self.origin_result_frame.pack(side="left", expand=True, padx=20, pady=20)

        self.origin_result_title = ctk.CTkLabel(
            self.origin_result_frame, text="원본 모델 추론", font=("Arial", 16, "bold")
        )
        self.origin_result_title.pack(pady=10)

        self.origin_progressbar = ctk.CTkProgressBar(
            self.origin_result_frame, width=400
        )
        self.origin_progressbar.set(0)  # 0 ~ 1
        self.origin_progressbar.pack(pady=10)

        self.origin_time_label = ctk.CTkLabel(
            self.origin_result_frame, text="소요 시간: -"
        )
        self.origin_time_label.pack(pady=10)

        # 오른쪽 결과 (convert model)
        self.convert_result_frame = ctk.CTkFrame(self.result_frame)
        self.convert_result_frame.pack(side="left", expand=True, padx=20, pady=20)

        self.convert_result_title = ctk.CTkLabel(
            self.convert_result_frame, text="변환 모델 추론", font=("Arial", 16, "bold")
        )
        self.convert_result_title.pack(pady=10)

        self.convert_progressbar = ctk.CTkProgressBar(
            self.convert_result_frame, width=400
        )
        self.convert_progressbar.set(0)
        self.convert_progressbar.pack(pady=10)

        self.convert_time_label = ctk.CTkLabel(
            self.convert_result_frame, text="소요 시간: -"
        )
        self.convert_time_label.pack(pady=10)

        # ---------------- 하단: 시작 버튼 ----------------
        self.start_button = ctk.CTkButton(
            self,
            text="시작",
            width=160,
            height=50,
            command=self.start_process,
            fg_color="gray",  # 기본은 회색
            hover_color="gray",
            state="disabled",  # 기본 비활성화
        )
        self.start_button.pack(side="bottom", pady=20)

    def get_size_readable(self, path):
        """파일 또는 폴더 전체 크기를 사람이 읽기 쉬운 형태로 반환"""
        total_size = 0

        if os.path.isfile(path):
            total_size = os.path.getsize(path)
        else:
            # 폴더의 모든 파일 크기 합산
            for root, dirs, files in os.walk(path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        total_size += os.path.getsize(fp)
                    except:
                        pass

        # 사람이 읽기 쉬운 형태로 변환
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if total_size < 1024:
                return f"{total_size:.2f} {unit}"
            total_size /= 1024

    # 공통 파일/폴더 선택
    def open_path_dialog(self):
        path = filedialog.askopenfilename(
            title="파일을 선택하거나 취소 후 폴더를 선택하세요",
            filetypes=[("All files", "*.*")],
        )

        if not path:
            path = filedialog.askdirectory(title="폴더 선택")

        if not path:
            return None
        return path

    # 왼쪽 버튼: origin 모델 선택
    def select_origin_model(self):
        path = self.open_path_dialog()
        if path:
            self.origin_model_path = path
            print(f"origin_model_path = {self.origin_model_path}")
            self.update_origin_label()
            self.update_start_button_state()

    # 오른쪽 버튼: convert 모델 선택
    def select_convert_model(self):
        path = self.open_path_dialog()
        if path:
            self.convert_model_path = path
            print(f"convert_model_path = {self.convert_model_path}")
            self.update_convert_label()
            self.update_start_button_state()

    # 라벨 업데이트
    def update_origin_label(self):
        size_text = self.get_size_readable(self.origin_model_path)
        self.origin_label.configure(
            text=f"선택된 경로:\n{self.origin_model_path}\n크기: {size_text}"
        )

    def update_convert_label(self):
        size_text = self.get_size_readable(self.convert_model_path)
        self.convert_label.configure(
            text=f"선택된 경로:\n{self.convert_model_path}\n크기: {size_text}"
        )

    # 시작 버튼 상태/색상 업데이트
    def update_start_button_state(self):
        if self.origin_model_path and self.convert_model_path:
            self.start_button.configure(
                state="normal", fg_color="red", hover_color="#aa0000"
            )
        else:
            self.start_button.configure(
                state="disabled", fg_color="gray", hover_color="gray"
            )

    # 시작 버튼 눌렀을 때 동작
    def start_process(self):
        print("=== START ===")
        print("origin_model_path :", self.origin_model_path)
        print("convert_model_path:", self.convert_model_path)

        # 비디오 존재 여부 체크
        if not os.path.exists(self.video_path):
            print(f"[ERROR] video_path 가 존재하지 않습니다: {self.video_path}")
            return

        # 모델 로드 (이미 로드된 상태면 재사용)
        if self.origin_model is None:
            print("원본 모델 로딩 중...")
            self.origin_model = YOLO(self.origin_model_path)

        if self.convert_model is None:
            print("변환 모델 로딩 중...")
            self.convert_model = YOLO(self.convert_model_path)

        # 진행 전 UI 초기화
        self.origin_progressbar.set(0)
        self.convert_progressbar.set(0)
        self.origin_time_label.configure(text="소요 시간: -")
        self.convert_time_label.configure(text="소요 시간: -")

        # 시작 버튼 잠금
        self.start_button.configure(
            state="disabled", fg_color="gray", hover_color="gray"
        )

        # 별도 스레드에서 실제 추론 돌리기
        t = threading.Thread(target=self.run_benchmark_thread, daemon=True)
        t.start()

    # 백그라운드 스레드에서 origin -> convert 순서로 실행
    def run_benchmark_thread(self):
        # 1) origin model
        origin_time = self.run_video_inference(self.origin_model, "origin")

        # 시간 표시 업데이트
        self.after(
            0,
            lambda: self.origin_time_label.configure(
                text=f"소요 시간: {origin_time:.2f} 초"
            ),
        )

        # 2) convert model
        convert_time = self.run_video_inference(self.convert_model, "convert")

        self.after(
            0,
            lambda: self.convert_time_label.configure(
                text=f"소요 시간: {convert_time:.2f} 초"
            ),
        )

        # 모두 끝난 뒤 시작 버튼 다시 활성화
        self.after(
            0,
            lambda: self.start_button.configure(
                state="normal", fg_color="red", hover_color="#aa0000"
            ),
        )

    # 비디오 전체를 한 번 도는 추론 함수 (시각화 X, progressbar만 업데이트)
    def run_video_inference(self, model: YOLO, which: str) -> float:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] 비디오를 열 수 없습니다: {self.video_path}")
            return 0.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            # 프레임 수를 가져오지 못하면 수동 카운트
            total_frames = 0
            temp_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                temp_frames.append(frame)
                total_frames += 1
            cap.release()
            # 다시 열어서 재사용
            cap = cv2.VideoCapture(self.video_path)
        print(f"[{which}] total_frames = {total_frames}")

        processed = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 추론 (시각화 X)
            _ = model.predict(source=frame, verbose=False)

            processed += 1
            if total_frames > 0:
                progress = processed / total_frames
            else:
                progress = 0

            # progressbar 업데이트 (메인 스레드에서)
            if which == "origin":
                self.after(0, lambda v=progress: self.origin_progressbar.set(v))
            else:
                self.after(0, lambda v=progress: self.convert_progressbar.set(v))

        cap.release()
        end_time = time.time()
        elapsed = end_time - start_time

        # 최종적으로 100%로 맞춰주기
        if which == "origin":
            self.after(0, lambda: self.origin_progressbar.set(1.0))
        else:
            self.after(0, lambda: self.convert_progressbar.set(1.0))

        print(f"[{which}] elapsed: {elapsed:.2f} sec")
        return elapsed
