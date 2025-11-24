from ultralytics import YOLO
from tkinter import filedialog  # 파일/폴더 선택창
from PIL import Image

import os
import customtkinter as ctk
import cv2


class YOLOPredictorGUI(ctk.CTk):
    origin_model_path = None
    convert_model_path = None

    origin_model: YOLO | None = None
    convert_model: YOLO | None = None

    image_path = os.path.join(
        "datasets", "image", "test.jpg"
    )  # 예측에 사용할 이미지 경로

    def __init__(self):
        super().__init__()
        self.title("YOLO Model Valid")
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

        # ---------------- 중단: 결과 이미지 표시 영역 ----------------
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.pack(fill="both", expand=True, pady=10)

        # 왼쪽 결과 (origin model)
        self.origin_result_label = ctk.CTkLabel(
            self.result_frame, text="원본 모델 결과", width=500, height=400
        )
        self.origin_result_label.pack(side="left", expand=True, padx=10, pady=10)

        # 오른쪽 결과 (convert model)
        self.convert_result_label = ctk.CTkLabel(
            self.result_frame, text="변환 모델 결과", width=500, height=400
        )
        self.convert_result_label.pack(side="left", expand=True, padx=10, pady=10)

        # CTkImage 를 참조로 들고 있어야 이미지가 안 사라짐
        self.origin_ctk_image = None
        self.convert_ctk_image = None

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
        # 하단 가운데 정렬
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
            # 둘 다 선택됨 → 빨간색 + 활성화
            self.start_button.configure(
                state="normal", fg_color="red", hover_color="#aa0000"
            )
        else:
            # 하나라도 비어 있음 → 회색 + 비활성화
            self.start_button.configure(
                state="disabled", fg_color="gray", hover_color="gray"
            )

    # 시작 버튼 눌렀을 때 동작
    def start_process(self):
        print("=== START ===")
        print("origin_model_path :", self.origin_model_path)
        print("convert_model_path:", self.convert_model_path)

        # 이미지 존재 여부 체크
        if not os.path.exists(self.image_path):
            print(f"[ERROR] image_path 가 존재하지 않습니다: {self.image_path}")
            return

        # 모델 로드 (이미 로드된 상태면 재사용)
        if self.origin_model is None:
            print("원본 모델 로딩 중...")
            self.origin_model = YOLO(self.origin_model_path)

        if self.convert_model is None:
            print("변환 모델 로딩 중...")
            self.convert_model = YOLO(self.convert_model_path)

        # 예측 수행
        print("원본 모델 예측 중...")
        origin_results = self.origin_model.predict(
            self.image_path, verbose=False, conf=0.5
        )[0]
        origin_annotated = origin_results.plot()  # numpy (BGR)

        print("변환 모델 예측 중...")
        convert_results = self.convert_model.predict(
            self.image_path, verbose=False, conf=0.4
        )[0]
        convert_annotated = convert_results.plot()  # numpy (BGR)

        # 결과 이미지를 GUI에 표시
        self.show_result_images(origin_annotated, convert_annotated)

    # 결과 이미지를 Tkinter에 표시
    def show_result_images(self, origin_np, convert_np):
        # BGR -> RGB 변환
        origin_rgb = cv2.cvtColor(origin_np, cv2.COLOR_BGR2RGB)
        convert_rgb = cv2.cvtColor(convert_np, cv2.COLOR_BGR2RGB)

        origin_pil = Image.fromarray(origin_rgb)
        convert_pil = Image.fromarray(convert_rgb)

        # 창에 맞게 대략 리사이즈 (비율 유지)
        max_size = (500, 500)
        origin_pil.thumbnail(max_size)
        convert_pil.thumbnail(max_size)

        # CTkImage 로 변환해서 Label에 부착
        self.origin_ctk_image = ctk.CTkImage(
            light_image=origin_pil, size=origin_pil.size
        )
        self.convert_ctk_image = ctk.CTkImage(
            light_image=convert_pil, size=convert_pil.size
        )

        self.origin_result_label.configure(image=self.origin_ctk_image, text="")
        self.convert_result_label.configure(image=self.convert_ctk_image, text="")
