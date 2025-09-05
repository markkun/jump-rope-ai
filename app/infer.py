# app/infer.py
"""
端到端跳绳分析脚本（带日志、异常处理、性能监控）
支持摄像头、本地视频、RTSP 流
"""
import cv2
import argparse
import logging
import time
import os
from pathlib import Path
from src.pipeline import JumpRopePipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('logs/infer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Jump Rope AI - End-to-End Inference")
    parser.add_argument('--source', type=str, default='0', help='视频源：0（摄像头）、文件路径、RTSP地址')
    parser.add_argument('--stgcn-ckpt', type=str, default='models/stgcn_best.pth', help='ST-GCN 模型路径')
    parser.add_argument('--output', type=str, default=None, help='输出视频路径（如 result.mp4）')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='推理设备')
    parser.add_argument('--show', action='store_true', help='是否显示画面')
    parser.add_argument('--log-fps', action='store_true', help='打印处理帧率')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    logger.info(f"启动跳绳 AI 分析系统 | 源: {args.source} | 设备: {args.device}")

    # 处理视频源
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error(f"无法打开视频源: {source}")
        return

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"视频流信息: {w}x{h} @ {fps}fps")

    # 初始化流水线
    try:
        pipeline = JumpRopePipeline(stgcn_ckpt=args.stgcn_ckpt, device=args.device)
        logger.info("✅ JumpRopePipeline 初始化成功")
    except Exception as e:
        logger.error(f"初始化 pipeline 失败: {e}")
        return

    # 视频写入器
    out = None
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
        logger.info(f"✅ 视频写入器已启动: {args.output}")

    # 性能监控
    frame_count = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("视频流结束或读取失败")
                break

            frame_count += 1

            # 🔥 核心推理
            infer_start = time.time()
            results = pipeline.infer_frame(frame)
            infer_time = time.time() - infer_start

            # 可视化
            for res in results:
                if res['action'] == 'jumping':
                    x1, y1, x2, y2 = map(int, res['bbox'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{res['id']} Count:{res['count']} Score:{res['score']:.1f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 显示帧率
            cv2.putText(frame, f"FPS: {1/infer_time:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 显示或写入
            if args.show:
                cv2.imshow('Jump Rope AI', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            if out:
                out.write(frame)

            # 日志
            if args.log_fps and frame_count % 30 == 0:
                elapsed = time.time() - start_time
                logger.info(f"处理 {frame_count} 帧 | 平均 FPS: {frame_count / elapsed:.1f}")

    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"运行时错误: {e}", exc_info=True)
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        logger.info(f"✅ 推理完成 | 处理 {frame_count} 帧 | 平均 FPS: {frame_count / (time.time() - start_time):.1f}")

if __name__ == "__main__":
    main()
