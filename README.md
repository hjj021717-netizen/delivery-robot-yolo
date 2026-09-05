# Delivery Robot YOLO Detection

## Classes
- traffic_light_red
- traffic_light_green
- crosswalk
- temi

## Usage
```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model('image.jpg', conf=0.6)
results[0].show()
```
