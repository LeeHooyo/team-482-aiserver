import cv2
import numpy as np
import time
from flask import Flask, jsonify, request
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
from threading import Timer
import boto3

app = Flask(__name__)

# S3 클라이언트 생성
s3_client = boto3.client('s3')
BUCKET_NAME = 'team486-cctvvideo'

spf_values = {
    "spfA": None,
    "spfB": None,
    "spfC": None,
    "spfD": None,
    "spfE": None,
    "spfF": None
}

accident_status = {
    "A": {"isAccident": False, "accidentType": None},
    "B": {"isAccident": False, "accidentType": None},
    "C": {"isAccident": False, "accidentType": None},
    "D": {"isAccident": False, "accidentType": None},
    "E": {"isAccident": False, "accidentType": None},
    "F": {"isAccident": False, "accidentType": None}
}

accident_videos = {
    "crash1.mp4": "CAR_TO_CAR LEFT_TO_RIGHT",
    "crash2.mp4": "CAR_TO_CAR UP_TO_DOWN",
    "crash3.mp4": "CAR_TO_OBJECT LEFT_TO_RIGHT",
    "crash4.mp4": "CAR_TO_OBJECT UP_TO_DOWN"
}

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

def reset_accident_flags(key):
    accident_status[key]["isAccident"] = False
    accident_status[key]["accidentType"] = None

def download_video_from_s3(video_key):
    local_path = f"/tmp/{os.path.basename(video_key)}"
    s3_client.download_file(BUCKET_NAME, video_key, local_path)
    return local_path

def process_video(video_key, spf_key):
    local_video_path = download_video_from_s3(video_key)
    model = YOLO("yolov8l.pt")

    while True:  # 무한 루프를 통해 영상 반복 재생
        cap = cv2.VideoCapture(local_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        target_classes = ["car", "truck", "bicycle", "motorcycle", "bus"]

        total_vehicles = 0
        up_dir_vehicles = 0
        left_dir_vehicles = 0
        previous_up_dir_vehicles = 0
        previous_left_dir_vehicles = 0
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
                road_key = spf_key.replace('spf', '')
                accident_status[road_key]["isAccident"] = True
                accident_status[road_key]["accidentType"] = accident_videos[accident_video]

                cap.release()
                accident_cap = cv2.VideoCapture(os.path.join(os.getcwd(), accident_video))
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
                                # Assume crash video has vehicles spread naturally between up and left directions
                                if box[1] < frame_height / 2:  # Up direction
                                    up_dir_vehicles += 1
                                else:  # Left direction
                                    left_dir_vehicles += 1

                accident_cap.release()
                Timer(180, reset_accident_flags, [road_key]).start()

            if not accident_occurred:
                results = model(frame)
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    class_ids_detected = result.boxes.cls.cpu().numpy()
                    for i, box in enumerate(boxes):
                        class_id = int(class_ids_detected[i])
                        class_name = model.names[class_id]
                        if class_name in target_classes:
                            total_vehicles += 1
                            # Check if the vehicle is in the Up-Dir or Left-Dir based on y-coordinate
                            if box[1] < frame_height / 2:  # Up direction
                                up_dir_vehicles += 1
                            else:  # Left direction
                                left_dir_vehicles += 1

            current_frame += 1

            current_time = time.time()
            if current_time - start_time >= 5:  # 5초마다 한번씩 SPF 계산
                # Adjust up_dir and left_dir to ensure natural changes
                if up_dir_vehicles > previous_up_dir_vehicles:
                    up_dir_vehicles = min(up_dir_vehicles, previous_up_dir_vehicles + 5)  # limit increase to 5
                else:
                    up_dir_vehicles = max(up_dir_vehicles, previous_up_dir_vehicles - 5)  # limit decrease to 5

                if left_dir_vehicles > previous_left_dir_vehicles:
                    left_dir_vehicles = min(left_dir_vehicles, previous_left_dir_vehicles + 5)  # limit increase to 5
                else:
                    left_dir_vehicles = max(left_dir_vehicles, previous_left_dir_vehicles - 5)  # limit decrease to 5

                aadt = (total_vehicles / (current_time - start_time)) * 86400  # 일일 평균 교통량 계산
                spf_value = calculate_spf(aadt)

                normalized_spf_value = normalize_spf(spf_value, spf_min=0, spf_max=calculate_spf(50 * 86400))

                spf_values[spf_key] = {
                    "congestion": normalized_spf_value,
                    "up_dir": up_dir_vehicles,
                    "left_dir": left_dir_vehicles
                }

                previous_up_dir_vehicles = up_dir_vehicles
                previous_left_dir_vehicles = left_dir_vehicles
                total_vehicles = 0
                up_dir_vehicles = 0
                left_dir_vehicles = 0
                start_time = current_time

        cap.release()

@app.route('/process_videos', methods=['POST'])
def process_videos():
    video_keys = request.json.get('video_keys')  # S3 비디오 파일 키 목록
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
    data = {}
    for key in spf_values.keys():
        road_key = key.replace('spf', '')
        data[key] = {
            "congestion": spf_values[key]["congestion"],
            "up_dir": spf_values[key]["up_dir"],
            "left_dir": spf_values[key]["left_dir"],
            "radius": 1.8 if road_key != 'A' else 1.6,
            "isAccident": accident_status[road_key]["isAccident"],
            "accidentType": accident_status[road_key]["accidentType"]
        }
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)