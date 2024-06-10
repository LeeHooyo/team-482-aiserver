import cv2
import numpy as np
import time
from flask import Flask, jsonify, request
from ultralytics import YOLO
from threading import Timer
from deep_sort_realtime.deepsort_tracker import DeepSort
import random  # random 모듈 임포트

app = Flask(__name__)

spf_values = {
    "radius": 0.2,
    "congestion": 0,
    "isAccident": False,
    "accidentType": None,
    "numOfcar": 0
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

def reset_accident_flags():
    spf_values["isAccident"] = False
    spf_values["accidentType"] = None

def process_video(video_name):
    local_video_path = video_name
    model = YOLO(model_path)
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
            spf_values["isAccident"] = True
            spf_values["accidentType"] = accident_type

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
            Timer(180, reset_accident_flags).start()

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
            spf_values["congestion"] = normalized_spf_value
            spf_values["numOfcar"] = len(tracked_vehicles)

            total_vehicles = 0
            tracked_vehicles = set()
            start_time = current_time

    cap.release()

@app.route('/process_video', methods=['POST'])
def process_video_route():
    video_name = request.json.get('video_name')  # 로컬 비디오 파일 이름
    if not video_name:
        return jsonify({"error": "No video name provided"}), 400
    
    try:
        process_video(video_name)
        return jsonify({"status": "Video processed successfully"})
    except Exception as exc:
        return jsonify({"error": f"Video processing generated an exception: {exc}"}), 500

@app.route('/get_spf', methods=['GET'])
def get_spf():
    return jsonify(spf_values)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
