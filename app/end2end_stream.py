# app/end2end_stream.py
"""
跳绳 AI + RTMP 推流系统
输入：摄像头/视频
输出：分析后画面推送到 RTMP 服务器（如 OBS、Nginx）
"""
import cv2
import subprocess
import argparse
import logging
from pathlib import Path
from src.pipeline import JumpRopePipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.FileHandler('logs/stream.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Jump Rope AI - RTMP Stream Output")
    parser.add_argument('--source', type=str, default='0', help='输入源')
    parser.add_argument('--stgcn-ckpt', type=str, default='models/stgcn_best.pth')
    parser.add_argument('--rtmp-addr', type=str, required=True, help='RTMP 推流地址，如 rtmp://localhost/live/stream')
    parser.add_argument('--fps', type=int, default=25, help='推流帧率')
    parser.add_argument('--preset', type=str, default='ultrafast', choices=['ultrafast', 'superfast', 'veryfast'], help='FFmpeg 编码速度')
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"启动 RTMP 推流系统 | 源: {args.source} | 推流地址: {args.rtmp_addr}")

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if not cap.isOpened():
        logger.error(f"无法打开视频源: {args.source}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # FFmpeg 推流命令
    command = [
        'ffmpeg',
        '-y',  # 覆盖输出
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{w}x{h}",
        '-r', str(args.fps),
        '-i', '-',  # 从 stdin 读取
        '-f', 'flv',
        '-c:v', 'libx264',
        '-preset', args.preset,
        '-b:v', '2048k',
        '-pix_fmt', 'yuv420p',
        '-an',  # 无音频
        args.rtmp_addr
    ]

    logger.info(f"FFmpeg 命令: {' '.join(command)}")
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    # 初始化 pipeline
    pipeline = JumpRopePipeline(stgcn_ckpt=args.stgcn_ckpt)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 推理
            results = pipeline.infer_frame(frame)

            # 可视化
            for res in results:
                if res['action'] == 'jumping':
                    x1, y1, x2, y2 = map(int, res['bbox'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{res['id']} Count:{res['count']} Score:{res['score']:.1f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 推送到 FFmpeg
            process.stdin.write(frame.tobytes())

    except KeyboardInterrupt:
        logger.info("用户中断推流")
    except Exception as e:
        logger.error(f"推流错误: {e}", exc_info=True)
    finally:
        cap.release()
        if process.stdin:
            process.stdin.close()
        process.terminate()
        logger.info("✅ RTMP 推流结束")

if __name__ == "__main__":
    main()
