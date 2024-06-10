import cv2
import numpy as np
import time
from flask import Flask, jsonify, request
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
from threading import Timer
from deep_sort_realtime.deepsort_tracker import DeepSort

app = Flask(__name__)

spf_values = {
    "spfA": {"radius": 0.2, "congestion": 0, "isAccident": False, "accidentType": None, "numOfcar": 0},
    "spfB": {"radius": 0.2, "congestion": 0, "isAccident": False, "accidentType": None, "numOfcar": 0},
    "spfC": {"radius": 0.2, "congestion": 0, "isAccident": False, "accidentType": None, "numOfcar": 0},
    "spfD": {"radius": 0.2, "congestion": 0, "isAccident": False, "accidentType": None, "numOfcar": 0},
    "spfE": {"radius": 0.2, "congestion": 0, "isAccident": False, "accidentType": None, "numOfcar": 0},
    "spfF": {"radius": 0.2, "congestion": 0, "isAccident": False, "accidentType": None, "numOfcar": 0},
}

accident_videos = {
    "crash1.mp4": "CAR_TO_CAR",
    "crash2.mp4": "CAR_TO_CAR",
    "crash3.mp4": "CAR_TO_HUMAN",
    "crash4.mp4": "CAR_TO_HUMAN",
}

model_path = "yolov8l.pt"
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.7)

# Safety Performance Function (SPF)
def calculate_spf(aadt):
    A = 0.001
    B = 0.5
    return A * (aadt ** B)

# SPF 값을 0에서 50 사이로 변환하는 함수
def normalize_spf(spf_value, spf_min=0, spf_max=calculate_spf(50 * 86400), target_min=0, target_max=50):
    if spf_max == spf_min:
        return target_min
    normalized_value = (spf_value - spf_min) / (spf_max - spf_min) * (target_max - target_min) + target_min
    return normalized_value

def reset_accident_flags(spf_key):
    spf_values[spf_key]["isAccident"] = False
    spf_values[spf_key]["accidentType"] = None

def process_video(video_name, spf_key):
    local_video_path = video_name
    model = YOLO(model_path)

    while True:  # 무한 루프를 통해 영상 반복 재생
        cap = cv2.VideoCapture(local_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        target_classes = ["car", "truck", "bicycle", "motorcycle", "bus"]

        total_vehicles = 0
        tracked_vehicles = set()
        start_time = time.time()
        current_frame = 0

        accident_occurred = False
        accident_start_frame = random.randint(0, total_frames - 1) if random.random() < 0.5 else None
        accident_video = random.choice(list(accident_videos.keys())) if accident_start_frame else None

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if accident_start_frame and current_frame == accident_start_frame:
                accident_occurred = True
                accident_type = accident_videos[accident_video]
                spf_values[spf_key]["isAccident"] = True
                spf_values[spf_key]["accidentType"] = accident_type

                cap.release()
                accident_cap = cv2.VideoCapture(accident_video)
                while accident_cap.isOpened():
                    ret, accident_frame = accident_cap.read()
                    if not ret:
                        break
                    results = model(accident_frame)
                    for result in results:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        class_ids_detected = result.boxes.cls.cpu().numpy()
                        for i, box in enumerate(boxes):
                            class_id = int(class_ids_detected[i])
                            class_name = model.names[class_id]
                            if class_name in target_classes:
                                total_vehicles += 1

                accident_cap.release()
                Timer(180, reset_accident_flags, [spf_key]).start()

            if not accident_occurred:
                results = model(frame)
                detections = []
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    class_ids_detected = result.boxes.cls.cpu().numpy()
                    for i, box in enumerate(boxes):
                        class_id = int(class_ids_detected[i])
                        class_name = model.names[class_id]
                        if class_name in target_classes:
                            detections.append((box, class_id))

                tracks = tracker.update_tracks(detections, frame=frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    if track_id not in tracked_vehicles:
                        tracked_vehicles.add(track_id)
                        total_vehicles += 1

            current_frame += 1

            current_time = time.time()
            if current_time - start_time >= 5:  # 5초마다 한번씩 SPF 계산
                aadt = (total_vehicles / (current_time - start_time)) * 86400  # 일일 평균 교통량 계산
                spf_value = calculate_spf(aadt)
                normalized_spf_value = normalize_spf(spf_value, spf_min=0, spf_max=calculate_spf(50 * 86400))
                spf_values[spf_key]["congestion"] = normalized_spf_value
                spf_values[spf_key]["numOfcar"] = len(tracked_vehicles)

                total_vehicles = 0
                tracked_vehicles = set()
                start_time = current_time

        cap.release()

@app.route('/process_videos', methods=['POST'])
def process_videos():
    video_keys = request.json.get('video_keys')  # 로컬 비디오 파일 키 목록
    video_keys_map = ['spfA', 'spfB', 'spfC', 'spfD', 'spfE', 'spfF']

    with ThreadPoolExecutor(max_workers=len(video_keys)) as executor:
        future_to_video = {
            executor.submit(process_video, video_key, video_keys_map[idx]): video_keys_map[idx]
            for idx, video_key in enumerate(video_keys)
        }
        for future in as_completed(future_to_video):
            video_key = future_to_video[future]
            try:
                future.result()
            except Exception as exc:
                return jsonify({"error": f"Video {video_key} generated an exception: {exc}"}), 500

    return jsonify({"status": "Videos are being processed"})

@app.route('/get_spf', methods=['GET'])
def get_spf():
    data = []
    for spf_key, spf_value in spf_values.items():
        spf_data = {
            "id": spf_key,
            "radius": spf_value["radius"],
            "congestion": spf_value["congestion"],
            "isAccident": spf_value["isAccident"],
            "accidentType": spf_value["accidentType"],
            "numOfcar": spf_value["numOfcar"]
        }
        data.append(spf_data)
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
