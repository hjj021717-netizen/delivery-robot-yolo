from collections import deque, Counter
import time
import cv2
import numpy as np
import logging

logging.basicConfig(
    filename='robot_behavior.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


class TrafficLightColorDetector:
    """
    YOLO가 찾은 신호등 박스 안에서 빨간색 여부만 판단.
    빨간색 감지 정확도가 높다는 실측 결과를 반영해,
    빨간색이 아니면 무조건 초록불로 간주.
    """
    def __init__(self, red_threshold=0.15, min_bright_pixels=5):
        self.red_threshold = red_threshold
        self.min_bright_pixels = min_bright_pixels

    def detect_color(self, image, box_xyxy, top_percent=0.15):
        x1, y1, x2, y2 = map(int, box_xyxy)
        light_region = image[y1:y2, x1:x2]

        gray = cv2.cvtColor(light_region, cv2.COLOR_BGR2GRAY)
        if gray.size == 0:
            return "UNKNOWN", 0

        threshold = np.percentile(gray, 100 - top_percent * 100)
        bright_mask = gray >= threshold
        bright_pixels = light_region[bright_mask]

        if len(bright_pixels) < self.min_bright_pixels:
            return "UNKNOWN", 0

        b = bright_pixels[:, 0].astype(float)
        g = bright_pixels[:, 1].astype(float)
        r = bright_pixels[:, 2].astype(float)
        total = r + g + b + 1e-6
        red_score = np.mean((r - np.maximum(g, b)) / total)

        if red_score >= self.red_threshold:
            return "RED", red_score
        else:
            return "GREEN", 1 - red_score


class TrafficLightController:
    """
    신호등 색깔만 보고 로봇을 멈추거나 보내는 단순한 로직.
    - 빨간불 또는 미감지/불안정: 정지 (/cmd_vel 0)
    - 초록불: 진행 (Nav2가 이미 그은 웨이포인트대로 이동, 별도 명령 불필요)
    - 거리/장애물(temi, 보행자) 판단 없음 - Nav2가 웨이포인트대로 이동하며 처리.
    """
    def __init__(self, history_size=5, min_confidence=0.5):
        self.state = "MOVING"
        self.history_size = history_size
        self.min_confidence = min_confidence
        self.detection_history = deque(maxlen=history_size)

    def filter_low_confidence(self, detections):
        return [d for d in detections if d['conf'] >= self.min_confidence]

    def get_stable_light_color(self, current_detections):
        filtered = self.filter_low_confidence(current_detections)
        traffic_lights = [d for d in filtered if 'traffic_light' in d['class']]

        if traffic_lights:
            best = max(traffic_lights, key=lambda x: x['conf'])
            self.detection_history.append(best['class'])
        else:
            self.detection_history.append(None)

        votes = [v for v in self.detection_history if v is not None]
        if not votes:
            return None

        most_common, count = Counter(votes).most_common(1)[0]
        if count >= (self.history_size // 2 + 1):
            return most_common
        return "UNSTABLE"

    def decide_action(self, raw_detections):
        stable_light = self.get_stable_light_color(raw_detections)

        if stable_light is None or stable_light == "UNSTABLE":
            return self._stop("신호등 미감지 또는 불안정 - 안전 정지")

        if 'red' in str(stable_light):
            return self._stop("빨간불 감지")

        if 'green' in str(stable_light):
            return self._go("초록불 감지")

        return self._stop("알 수 없는 상태 - 안전 정지")

    def _stop(self, reason):
        self._log_and_print("STOP", reason)
        self.state = "STOPPED"
        return "STOP"

    def _go(self, reason):
        self._log_and_print("GO", reason)
        self.state = "MOVING"
        return "GO"

    def _log_and_print(self, action, reason):
        message = f"[{action}] {reason}"
        print(message)
        logging.info(message)


def yolo_results_to_detections(yolo_result, class_names, image, color_detector=None):
    """
    YOLO 예측 결과를 컨트롤러가 이해하는 형태로 변환.
    신호등이면 color_detector로 실제 색깔(빨간색 우선)을 재검증.
    """
    detections = []
    for box in yolo_result.boxes:
        class_id = int(box.cls[0])
        class_name = class_names[class_id]
        confidence = float(box.conf[0])

        if 'traffic_light' in class_name and color_detector is not None:
            box_xyxy = box.xyxy[0].cpu().numpy()
            verified_color, color_conf = color_detector.detect_color(image, box_xyxy)
            if verified_color == "RED":
                class_name = "traffic_light_red"
            elif verified_color == "GREEN":
                class_name = "traffic_light_green"
            else:
                continue

        detections.append({
            'class': class_name,
            'conf': confidence,
        })
    return detections
