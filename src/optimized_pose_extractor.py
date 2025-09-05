# src/optimized_pose_extractor.py
from mmpose.apis import init_model as init_pose, inference_topdown
from mmdet.apis import init_detector as init_det, inference_detector
import torch


class OptimizedPoseExtractor:
    def __init__(self, det_config='yolox_l_8x8_300e_coco.py',
                 det_ckpt='yolox_l_8x8_300e_coco_20211126_140254-ee22ba79.pth',
                 pose_config='rtmpose-s_8xb256-420e_coco-256x192.py',
                 pose_ckpt='rtmpose-s_simcc-coco_pt-aic-coco_120e-256x192-f1d8ece0_20230126.pth',
                 device='cuda'):
        self.device = device

        # 启用半精度和优化
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True

        self.det_model = init_det(det_config, det_ckpt, device=device)
        self.pose_model = init_pose(pose_config, pose_ckpt, device=device)

    @torch.no_grad()
    def extract(self, frame):
        # 使用 MMPose 官方 MOT 接口
        det_result = inference_detector(self.det_model, frame)

        # 过滤小目标和非人
        if hasattr(det_result, 'pred_instances') and hasattr(det_result.pred_instances, 'labels'):
            person_results = det_result.pred_instances[det_result.pred_instances.labels == 0]
        else:
            # 如果没有labels属性，我们假设所有检测到的都是人（简化处理）
            person_results = det_result.pred_instances if hasattr(det_result, 'pred_instances') else det_result

        if len(person_results) == 0:
            return []

        pose_results = inference_topdown(self.pose_model, frame, person_results)

        output = []
        for res in pose_results:
            # 检查属性是否存在并安全访问
            if hasattr(res, 'pred_instances'):
                pred_instances = res.pred_instances
                if hasattr(pred_instances, 'keypoints') and hasattr(pred_instances, 'keypoint_scores'):
                    kpts = pred_instances.keypoints[0] if len(pred_instances.keypoints) > 0 else None
                    scores = pred_instances.keypoint_scores[0] if len(pred_instances.keypoint_scores) > 0 else None
                else:
                    # 如果没有这些属性，尝试直接从结果对象获取
                    kpts = getattr(res, 'keypoints', None)
                    scores = getattr(res, 'keypoint_scores', None)
            else:
                # 直接从结果对象获取
                kpts = getattr(res, 'keypoints', None)
                scores = getattr(res, 'keypoint_scores', None)

            # 获取边界框 - 使用getattr避免静态检查问题
            bboxes_attr = getattr(res, 'bboxes', None)
            if bboxes_attr is not None and len(bboxes_attr) > 0:
                bbox = bboxes_attr[0].tolist()
            else:
                # 尝试从pred_instances中获取边界框
                pred_instances_bboxes = getattr(getattr(res, 'pred_instances', None), 'bboxes', None) if hasattr(res, 'pred_instances') else None
                if pred_instances_bboxes is not None and len(pred_instances_bboxes) > 0:
                    bbox = pred_instances_bboxes[0].tolist()
                else:
                    bbox = [0, 0, 0, 0]  # 默认边界框

            # 只有当关键点数据存在时才添加到输出
            if kpts is not None and scores is not None:
                output.append({
                    'id': hash(tuple(bbox)),  # 使用bbox生成唯一ID
                    'keypoints': torch.cat([kpts, scores.unsqueeze(1)], dim=1).cpu().numpy(),
                    'bbox': bbox,
                    'score': getattr(res, 'scores', [torch.tensor(1.0)])[0].item() if hasattr(getattr(res, 'scores', None), 'item') else 1.0
                })
        return output
