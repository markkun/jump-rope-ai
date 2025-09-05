# app/real_time.py（升级版）
from src.pipeline import JumpRopePipeline
import cv2

pipeline = JumpRopePipeline(stgcn_ckpt='models/checkpoints/stgcn_best.pth')

cap = cv2.VideoCapture(0)  # 摄像头
while True:
    ret, frame = cap.read()
    if not ret: break

    results = pipeline.infer_frame(frame)
    for res in results:
        if res['action'] == 'jumping':
            cv2.rectangle(frame, (int(res['bbox'][0]), int(res['bbox'][1])),
                          (int(res['bbox'][2]), int(res['bbox'][3])), (0,255,0), 2)
            cv2.putText(frame, f"ID:{res['id']} Count:{res['count']} Score:{res['score']:.1f}",
                        (int(res['bbox'][0]), int(res['bbox'][1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow('Jump Rope AI', frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
