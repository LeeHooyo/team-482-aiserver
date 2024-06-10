import cv2
import numpy as np
import time
from flask import Flask, jsonify, request
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
from threading import Timer

app = Flask(__name__)

spf_values = {
    "spfA": {
        "radius": 0.2,
        "congestion": 0,
        "directions": {
            "LEFT": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "RIGHT": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "UP": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "DOWN": {"numofcar": 0, "isAccident": False, "accidentType": None}
        }
    },
    "spfB": {
        "radius": 0.2,
        "congestion": 0,
        "directions": {
            "LEFT": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "RIGHT": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "UP": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "DOWN": {"numofcar": 0, "isAccident": False, "accidentType": None}
        }
    },
    "spfC": {
        "radius": 0.2,
        "congestion": 0,
        "directions": {
            "LEFT": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "RIGHT": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "UP": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "DOWN": {"numofcar": 0, "isAccident": False, "accidentType": None}
        }
    },
    "spfD": {
        "radius": 0.2,
        "congestion": 0,
        "directions": {
            "LEFT": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "RIGHT": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "UP": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "DOWN": {"numofcar": 0, "isAccident": False, "accidentType": None}
        }
    },
    "spfE": {
        "radius": 0.2,
        "congestion": 0,
        "directions": {
            "LEFT": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "RIGHT": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "UP": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "DOWN": {"numofcar": 0, "isAccident": False, "accidentType": None}
        }
    },
    "spfF": {
        "radius": 0.2,
        "congestion": 0,
        "directions": {
            "LEFT": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "RIGHT": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "UP": {"numofcar": 0, "isAccident": False, "accidentType": None},
            "DOWN": {"numofcar": 0, "isAccident": False, "accidentType": None}
        }
    }
}

accident_videos = {
    "crash1.mp4": "CAR_TO_CAR",
    "crash2.mp4": "CAR_TO_CAR",
    "crash3.mp4": "CAR_TO_HUMAN",
    "crash4.mp4": "CAR_TO_HUMAN"
}

model_path = 'yolov8l.pt'

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

def reset_accident_flags(spf_key, dir_key):
    spf_values[spf_key]["directions"][dir_key]["isAccident"] = False
    spf_values[spf_key]["directions"][dir_key]["accidentType"] = None

def process_video(video_name, spf_key):
    local_video_path = video_name
    model = YOLO(model_path)

    while True:  # 무한 루프를 통해 영상 반복 재생
        cap = cv2.VideoCapture(local_video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        target_classes = ["car", "truck", "bicycle", "motorcycle", "bus"]

        total_vehicles = 0
        dir_vehicles = {"LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0}
        previous_dir_vehicles = {"LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0}
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
                dir_key = random.choice(["LEFT", "RIGHT", "UP", "DOWN"])
                spf_values[spf_key]["directions"][dir_key]["isAccident"] = True
                spf_values[spf_key]["directions"][dir_key]["accidentType"] = accident_type

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
                                if box[0] < frame_width / 2:
                                    dir_vehicles["LEFT"] += 1
                                else:
                                    dir_vehicles["RIGHT"] += 1
                                if box[1] < frame_height / 2:
                                    dir_vehicles["UP"] += 1
                                else:
                                    dir_vehicles["DOWN"] += 1

                accident_cap.release()
                Timer(180, reset_accident_flags, [spf_key, dir_key]).start()

            if not accident_occurred:
                results = model(frame)
                current_frame_vehicles = set()
                for result in results:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    class_ids_detected = result.boxes.cls.cpu().numpy()
                    for i, box in enumerate(boxes):
                        class_id = int(class_ids_detected[i])
                        class_name = model.names[class_id]
                        if class_name in target_classes:
                            vehicle_id = (box[0], box[1], box[2], box[3])  # 차량의 위치로 고유 ID 생성
                            if vehicle_id not in current_frame_vehicles:
                                current_frame_vehicles.add(vehicle_id)
                                total_vehicles += 1
                                if box[0] < frame_width / 2:
                                    dir_vehicles["LEFT"] += 1
                                else:
                                    dir_vehicles["RIGHT"] += 1
                                if box[1] < frame_height / 2:
                                    dir_vehicles["UP"] += 1
                                else:
                                    dir_vehicles["DOWN"] += 1

            current_frame += 1

            current_time = time.time()
            if current_time - start_time >= 5:  # 5초마다 한번씩 SPF 계산
                aadt = (total_vehicles / (current_time - start_time)) * 86400  # 일일 평균 교통량 계산
                spf_value = calculate_spf(aadt)
                normalized_spf_value = normalize_spf(spf_value, spf_min=0, spf_max=calculate_spf(50 * 86400))
                spf_values[spf_key]["congestion"] = normalized_spf_value

                # 각 방향으로 차량 수 나누기
                total_num_of_cars = total_vehicles
                dir_vehicles_list = list(dir_vehicles.keys())

                # 차량 수를 자연스럽게 나누기
                remaining_cars = total_num_of_cars
                for dir_key in dir_vehicles_list[:-1]:
                    allocated_cars = int(total_num_of_cars * random.uniform(0.1, 0.3))
                    dir_vehicles[dir_key] = allocated_cars
                    remaining_cars -= allocated_cars

                dir_vehicles[dir_vehicles_list[-1]] = remaining_cars

                for dir_key in dir_vehicles:
                    spf_values[spf_key]["directions"][dir_key]["numofcar"] = dir_vehicles[dir_key]
                    previous_dir_vehicles[dir_key] = dir_vehicles[dir_key]

                total_vehicles = 0
                dir_vehicles = {"LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0}
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
            "directions": [
                {"dir": "LEFT", "numofcar": spf_value["directions"]["LEFT"]["numofcar"], "isAccident": spf_value["directions"]["LEFT"]["isAccident"], "accidentType": spf_value["directions"]["LEFT"]["accidentType"]},
                {"dir": "RIGHT", "numofcar": spf_value["directions"]["RIGHT"]["numofcar"], "isAccident": spf_value["directions"]["RIGHT"]["isAccident"], "accidentType": spf_value["directions"]["RIGHT"]["accidentType"]},
                {"dir": "UP", "numofcar": spf_value["directions"]["UP"]["numofcar"], "isAccident": spf_value["directions"]["UP"]["isAccident"], "accidentType": spf_value["directions"]["UP"]["accidentType"]},
                {"dir": "DOWN", "numofcar": spf_value["directions"]["DOWN"]["numofcar"], "isAccident": spf_value["directions"]["DOWN"]["isAccident"], "accidentType": spf_value["directions"]["DOWN"]["accidentType"]}
            ]
        }
        data.append(spf_data)
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
