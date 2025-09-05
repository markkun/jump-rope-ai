# app/infer.py
"""
端到端跳绳分析脚本（替代原 end2end.py）
支持视频文件或摄像头输入
"""
import cv2
import argparse
from src.pipeline import JumpRopePipeline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='视频路径或摄像头ID')
    parser.add_argument('--stgcn-ckpt', type=str, default='models/stgcn_best.pth')
    parser.add_argument('--output', type=str, default=None, help='输出视频路径')
    return parser.parse_args()

def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    # 初始化流水线
    pipeline = JumpRopePipeline(stgcn_ckpt=args.stgcn_ckpt)

    # 视频写入器（可选）
    if args.output:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 🔥 核心：端到端推理
        results = pipeline.infer_frame(frame)

        # 可视化
        for res in results:
            if res['action'] == 'jumping':
                x1, y1, x2, y2 = map(int, res['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{res['id']} Count:{res['count']} Score:{res['score']:.1f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Jump Rope AI', frame)
        if args.output:
            out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    if args.output:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
