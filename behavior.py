from collections import deque, Counter
import logging

logging.basicConfig(
    filename='robot_behavior.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

IGNORED_CLASSES = ["temi", "pedestrian"]


class TrafficLightController:
    """
    신호등 색깔만 보고 로봇을 멈추거나 보내는 로직.
    temi, pedestrian은 무시 (Nav2가 자체 처리).
    """
    def __init__(self, history_size=5, min_confidence=0.5):
        self.state = "MOVING"
        self.history_size = history_size
        self.min_confidence = min_confidence
        self.detection_history = deque(maxlen=history_size)

    def filter_low_confidence(self, detections):
        return [d for d in detections if d["conf"] >= self.min_confidence]

    def get_stable_light_color(self, current_detections):
        filtered = self.filter_low_confidence(current_detections)
        traffic_lights = [d for d in filtered if "traffic_light" in d["class"]]

        if traffic_lights:
            best = max(traffic_lights, key=lambda x: x["conf"])
            self.detection_history.append(best["class"])
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

        if "red" in str(stable_light):
            return self._stop("빨간불 감지")

        return self._go("빨간불 아님 - 초록불로 간주, 진행")

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
