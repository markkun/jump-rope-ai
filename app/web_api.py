from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)
pipeline = None  # 全局加载

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['video']
    video_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    video = cv2.imdecode(video_bytes, cv2.IMREAD_COLOR)
    
    results = []
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_results = pipeline.infer_frame(frame)
        results.extend(frame_results)
    cap.release()
    
    return jsonify(results)
