import cv2
import numpy as np
import time
from flask import Flask, jsonify, request
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
from threading import Timerimport cv2
import numpy as np
import time
from flask import Flask, jsonify, request
from ultralytics import YOLO
from threading import Timer
from deep_sort_realtime.deepsort_tracker import DeepSort
import random

app = Flask(__name__)

spf_values = {
    "spfA": {
        "radius": 0.2,
        "congestion": 0,
        "LEFT": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "RIGHT": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "UP": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "DOWN": {"numOfcar": 0, "isAccident": False, "accidentType": None},
    },
    "spfB": {
        "radius": 0.2,
        "congestion": 0,
        "LEFT": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "RIGHT": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "UP": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "DOWN": {"numOfcar": 0, "isAccident": False, "accidentType": None},
    },
    "spfC": {
        "radius": 0.2,
        "congestion": 0,
        "LEFT": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "RIGHT": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "UP": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "DOWN": {"numOfcar": 0, "isAccident": False, "accidentType": None},
    },
    "spfD": {
        "radius": 0.2,
        "congestion": 0,
        "LEFT": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "RIGHT": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "UP": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "DOWN": {"numOfcar": 0, "isAccident": False, "accidentType": None},
    },
    "spfE": {
        "radius": 0.2,
        "congestion": 0,
        "LEFT": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "RIGHT": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "UP": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "DOWN": {"numOfcar": 0, "isAccident": False, "accidentType": None},
    },
    "spfF": {
        "radius": 0.2,
        "congestion": 0,
        "LEFT": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "RIGHT": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "UP": {"numOfcar": 0, "isAccident": False, "accidentType": None},
        "DOWN": {"numOfcar": 0, "isAccident": False, "accidentType": None},
    }
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


def reset_accident_flags(spf_key, direction):
    spf_values[spf_key][direction]["isAccident"] = False
    spf_values[spf_key][direction]["accidentType"] = None


def process_video(video_name, spf_key):
    local_video_path = video_name
    model = YOLO(model_path)
    cap = cv2.VideoCapture(local_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    target_classes = ["car", "truck", "bicycle", "motorcycle", "bus"]

    total_vehicles = 0
    previous_dir_vehicles = {"LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0}
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

        frame_height, frame_width = frame.shape[:2]

        if accident_start_frame and current_frame == accident_start_frame:
            accident_occurred = True
            accident_type = accident_videos[accident_video]
            direction = random.choice(["LEFT", "RIGHT", "UP", "DOWN"])
            spf_values[spf_key][direction]["isAccident"] = True
            spf_values[spf_key][direction]["accidentType"] = accident_type

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
            Timer(180, reset_accident_flags, [spf_key, direction]).start()

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
            dir_vehicles = {"LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0}
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                if track_id not in tracked_vehicles:
                    tracked_vehicles.add(track_id)
                    total_vehicles += 1
                    # 네 방향으로 나누기
                    if bbox[0] < frame_width / 2:
                        if bbox[1] < frame_height / 2:
                            dir_vehicles["LEFT"] += 1
                        else:
                            dir_vehicles["DOWN"] += 1
                    else:
                        if bbox[1] < frame_height / 2:
                            dir_vehicles["UP"] += 1
                        else:
                            dir_vehicles["RIGHT"] += 1

            # 자연스러운 변화 적용
            for direction in ["LEFT", "RIGHT", "UP", "DOWN"]:
                spf_values[spf_key][direction]["numOfcar"] = max(previous_dir_vehicles[direction] - 5,
                                                                 min(previous_dir_vehicles[direction] + 5, dir_vehicles[direction]))
                previous_dir_vehicles[direction] = spf_values[spf_key][direction]["numOfcar"]

        current_frame += 1

        current_time = time.time()
        if current_time - start_time >= 5:  # 5초마다 한번씩 SPF 계산
            aadt = (total_vehicles / (current_time - start_time)) * 86400  # 일일 평균 교통량 계산
            spf_value = calculate_spf(aadt)
            normalized_spf_value = normalize_spf(spf_value, spf_min=0, spf_max=calculate_spf(50 * 86400))
            spf_values[spf_key]["congestion"] = normalized_spf_value

            total_vehicles = 0
            start_time = current_time

    cap.release()

@app.route('/process_videos', methods=['POST'])
def process_videos():
    video_keys = request.json.get('video_keys')  # 로컬 비디오 파일 키 목록
    video_keys_map = ['spfA', 'spfB', 'spfC', 'spfD', 'spfE', 'spfF']

    for idx, video_key in enumerate(video_keys):
        process_video(video_key, video_keys_map[idx])

    return jsonify({"status": "Videos are being processed"})

@app.route('/get_spf', methods=['GET'])
def get_spf():
    data = []
    for spf_key, spf_value in spf_values.items():
        spf_data = {
            "id": spf_key,
            "radius": spf_value["radius"],
            "congestion": spf_value["congestion"],
            "LEFT": spf_value["LEFT"],
            "RIGHT": spf_value["RIGHT"],
            "UP": spf_value["UP"],
            "DOWN": spf_value["DOWN"]
        }
        data.append(spf_data)
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

from deep_sort_realtime.deepsort_tracker import DeepSort

app = Flask(__name__)

spfA = {"radius": 0.2, "congestion": 0, "isAccident": False, "accidentType": None, "numOfcar": 0}
spfB = {"radius": 0.2, "congestion": 0, "isAccident": False, "accidentType": None, "numOfcar": 0}
spfC = {"radius": 0.2, "congestion": 0, "isAccident": False, "accidentType": None, "numOfcar": 0}
spfD = {"radius": 0.2, "congestion": 0, "isAccident": False, "accidentType": None, "numOfcar": 0}
spfE = {"radius": 0.2, "congestion": 0, "isAccident": False, "accidentType": None, "numOfcar": 0}
spfF = {"radius": 0.2, "congestion": 0, "isAccident": False, "accidentType": None, "numOfcar": 0}

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
    spf_key["isAccident"] = False
    spf_key["accidentType"] = None

def process_video(video_name, spf_key):
    local_video_path = video_name
    model = YOLO(model_path)

    while True:  # 무한 루프를 통해 영상 반복 재생
        cap = cv2.VideoCapture(local_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        target_classes = ["car", "truck", "bicycle", "motorcycle", "bus"]

        start_time = time.time()
        current_frame = 0

        accident_occurred = False
        accident_start_frame = random.randint(0, total_frames - 1) if random.random() < 0.5 else None
        accident_video = random.choice(list(accident_videos.keys())) if accident_start_frame else None

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            current_vehicles = 0  # 현재 프레임에서 감지된 차량 대수

            if accident_start_frame and current_frame == accident_start_frame:
                accident_occurred = True
                accident_type = accident_videos[accident_video]
                spf_key["isAccident"] = True
                spf_key["accidentType"] = accident_type

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
                                current_vehicles += 1  # 현재 프레임에서 감지된 차량 대수

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
                            current_vehicles += 1  # 현재 프레임에서 감지된 차량 대수

                tracks = tracker.update_tracks(detections, frame=frame)

            current_frame += 1

            current_time = time.time()
            if current_time - start_time >= 5:  # 5초마다 한번씩 SPF 계산
                aadt = (current_vehicles / (current_time - start_time)) * 86400  # 일일 평균 교통량 계산
                spf_value = calculate_spf(aadt)
                normalized_spf_value = normalize_spf(spf_value, spf_min=0, spf_max=calculate_spf(50 * 86400))
                spf_key["congestion"] = normalized_spf_value
                spf_key["numOfcar"] = current_vehicles  # 현재 프레임에서 감지된 차량 대수 업데이트

                current_vehicles = 0  # 현재 프레임에서 감지된 차량 대수 초기화
                start_time = current_time

        cap.release()

@app.route('/process_videos', methods=['POST'])
def process_videos():
    video_keys = request.json.get('video_keys')  # 로컬 비디오 파일 키 목록

    video_map = {
        "cctvA.mp4": spfA,
        "cctvB.mp4": spfB,
        "cctvC.mp4": spfC,
        "cctvD.mp4": spfD,
        "cctvE.mp4": spfE,
        "cctvF.mp4": spfF,
    }

    with ThreadPoolExecutor(max_workers=len(video_keys)) as executor:
        future_to_video = {
            executor.submit(process_video, video_key, video_map[video_key]): video_key
            for video_key in video_keys if video_key in video_map
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
    data = [
        {"id": "spfA", "radius": spfA["radius"], "congestion": spfA["congestion"], "isAccident": spfA["isAccident"], "accidentType": spfA["accidentType"], "numOfcar": spfA["numOfcar"]},
        {"id": "spfB", "radius": spfB["radius"], "congestion": spfB["congestion"], "isAccident": spfB["isAccident"], "accidentType": spfB["accidentType"], "numOfcar": spfB["numOfcar"]},
        {"id": "spfC", "radius": spfC["radius"], "congestion": spfC["congestion"], "isAccident": spfC["isAccident"], "accidentType": spfC["accidentType"], "numOfcar": spfC["numOfcar"]},
        {"id": "spfD", "radius": spfD["radius"], "congestion": spfD["congestion"], "isAccident": spfD["isAccident"], "accidentType": spfD["accidentType"], "numOfcar": spfD["numOfcar"]},
        {"id": "spfE", "radius": spfE["radius"], "congestion": spfE["congestion"], "isAccident": spfE["isAccident"], "accidentType": spfE["accidentType"], "numOfcar": spfE["numOfcar"]},
        {"id": "spfF", "radius": spfF["radius"], "congestion": spfF["congestion"], "isAccident": spfF["isAccident"], "accidentType": spfF["accidentType"], "numOfcar": spfF["numOfcar"]}
    ]
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
