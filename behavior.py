

import cv2
import numpy as np
from collections import deque, Counter
import time

class TrafficLightColorDetector:
    """
    YOLO가 찾은 신호등 박스 안에서, 정규화된 색상 점수(밝기 영향 최소화)로
    빨강/초록을 판단하고, 여러 프레임 다수결 + 타임아웃으로 안정화.
    """
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
