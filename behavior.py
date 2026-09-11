from collections import deque, Counter
import time
import math
import cv2
import numpy as np
import logging

logging.basicConfig(
    filename='robot_behavior.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

REAL_OBJECT_HEIGHTS_M = {
    'traffic_light_red': 0.30,
    'traffic_light_green': 0.30,
    'crosswalk': 0.50,
    'temi': 1.00,
    'pedestrian': 0.88,
}

CAMERA_FOV_VERTICAL_DEG = 55
IMAGE_HEIGHT_PIXELS = 640


def estimate_distance_from_box_size(box_height_pixels, class_name):
    if box_height_pixels <= 0:
        return 999
    real_height = REAL_OBJECT_HEIGHTS_M.get(class_name, 0.5)
    ratio = box_height_pixels / IMAGE_HEIGHT_PIXELS
    fov_rad = math.radians(CAMERA_FOV_VERTICAL_DEG)
    distance = real_height / (2 * ratio * math.tan(fov_rad / 2))
    return round(distance, 2)


def yolo_results_to_detections(yolo_result, class_names):
    detections = []
    for box in yolo_result.boxes:
        class_id = int(box.cls[0])
        class_name = class_names[class_id]
        confidence = float(box.conf[0])
        box_height = float(box.xywh[0][3])
        estimated_distance = estimate_distance_from_box_size(box_height, class_name)
        detections.append({
            'class': class_name,
            'conf': confidence,
            'distance': estimated_distance
        })
    return detections


class RobotBehaviorController:
    def __init__(self, history_size=5, min_confidence=0.5, api_client=None):
        self.state = "MOVING"
        self.green_light_entry_time = None
        self.history_size = history_size
        self.min_confidence = min_confidence
        self.detection_history = deque(maxlen=history_size)
        self.api_client = api_client

    def filter_low_confidence(self, detections):
        return [d for d in detections if d['conf'] >= self.min_confidence]

    def get_stable_state(self, current_detections):
        filtered = self.filter_low_confidence(current_detections)
        self.detection_history.append(filtered)
        light_votes = []
        for frame_detections in self.detection_history:
            traffic_lights = [d for d in frame_detections if 'traffic_light' in d['class']]
            if traffic_lights:
                best = max(traffic_lights, key=lambda x: x['conf'])
                light_votes.append(best['class'])
        if not light_votes:
            return None
        most_common = Counter(light_votes).most_common(1)[0]
        vote_class, vote_count = most_common
        if vote_count >= (self.history_size // 2 + 1):
            return vote_class
        return "UNSTABLE"

    def decide_action(self, raw_detections):
        filtered = self.filter_low_confidence(raw_detections)
        for det in filtered:
            if det['class'] == 'pedestrian' and det.get('distance', 999) < 0.2:
                return self._stop("보행자 20cm 이내 - 충돌 판정")
            if det['class'] == 'temi' and det.get('distance', 999) < 0.5:
                return self._stop("temi 근접")
        stable_light = self.get_stable_state(raw_detections)
        if stable_light is None:
            return self._continue("신호등 미감지 - 현재 상태 유지, 주의 진행")
        if stable_light == "UNSTABLE":
            return self._stop("신호등 색깔 불안정 - 안전을 위해 일시 정지")
        if stable_light == "traffic_light_red":
            return self._handle_red_light()
        if stable_light == "traffic_light_green":
            return self._handle_green_light()
        return self._continue("안전, 진행")

    def _handle_red_light(self):
        if self.state == "CROSSING":
            elapsed = time.time() - self.green_light_entry_time
            if elapsed > 3.0:
                return self._stop(f"빨간불 전환 후 {elapsed:.1f}초 경과 - 패널티 위험")
            return self._continue("빨간불이지만 3초 이내 - 횡단 계속 허용")
        return self._wait_at_light("빨간불 - 횡단보도 진입 대기")

    def _handle_green_light(self):
        if self.state == "WAITING_AT_LIGHT":
            self.green_light_entry_time = time.time()
            return self._go("초록불 - 횡단 시작")
        return self._continue("초록불 확인, 진행 유지")

    def _stop(self, reason):
        self._log_and_print("STOP", reason)
        self.state = "STOPPED"
        return "STOP"

    def _wait_at_light(self, reason):
        self._log_and_print("WAIT", reason)
        self.state = "WAITING_AT_LIGHT"
        return "STOP"

    def _go(self, reason):
        self._log_and_print("GO", reason)
        self.state = "CROSSING"
        return "GO"

    def _continue(self, reason=""):
        if reason:
            self._log_and_print("CONTINUE", reason)
        return "CONTINUE"

    def _log_and_print(self, action, reason):
        message = f"[{action}] {reason}"
        print(message)
        logging.info(message)


class TrafficLightColorDetector:
    def __init__(self, history_size=5, min_confidence=0.15, timeout_sec=0.3):
        self.history = deque(maxlen=history_size)
        self.min_confidence = min_confidence
        self.timeout_sec = timeout_sec
        self.last_detection_time = None

    def detect_color(self, image, box_xyxy):
        x1, y1, x2, y2 = map(int, box_xyxy)
        light_region = image[y1:y2, x1:x2]
        red_score, green_score = self._get_brightest_pixels_score(light_region)
        if abs(red_score - green_score) < 0.05:
            color, confidence = "UNKNOWN", 0
        elif red_score > green_score:
            color, confidence = "RED", red_score
        else:
            color, confidence = "GREEN", green_score
        self.history.append((color, confidence))
        self.last_detection_time = time.time()
        return self._get_stable_result()

    def _get_brightest_pixels_score(self, light_region, top_percent=0.15):
        gray = cv2.cvtColor(light_region, cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            return 0, 0
        threshold = np.percentile(gray, 100 - top_percent * 100)
        bright_mask = gray >= threshold
        bright_pixels = light_region[bright_mask]
        if len(bright_pixels) == 0:
            return 0, 0
        b = bright_pixels[:, 0].astype(float)
        g = bright_pixels[:, 1].astype(float)
        r = bright_pixels[:, 2].astype(float)
        total = r + g + b + 1e-6
        red_score = np.mean((r - np.maximum(g, b)) / total)
        green_score = np.mean((g - np.maximum(r, b)) / total)
        return red_score, green_score

    def _get_stable_result(self):
        if self.last_detection_time and (time.time() - self.last_detection_time > self.timeout_sec):
            return "UNKNOWN", 0
        colors = [c for c, conf in self.history if conf > self.min_confidence]
        if not colors:
            return "UNKNOWN", 0
        most_common_color, count = Counter(colors).most_common(1)[0]
        if count >= (len(self.history) // 2 + 1):
            avg_conf = np.mean([conf for c, conf in self.history if c == most_common_color])
            return most_common_color, avg_conf
        return "UNKNOWN", 0
