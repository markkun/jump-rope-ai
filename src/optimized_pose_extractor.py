# src/optimized_pose_extractor.py
from mmpose.apis import init_model as init_pose, inference_topdown
from mmdet.apis import init_model as init_det, inference_detector
from mmtrack.apis import inference_mot
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
        person_results = det_result.pred_instances[
            det_result.pred_instances.labels == 0]  # class 0 is person
        
        if len(person_results) == 0:
            return []

        pose_results = inference_topdown(self.pose_model, frame, person_results)

        output = []
        for res in pose_results:
            kpts = res.pred_instances.keypoints[0]
            scores = res.pred_instances.keypoint_scores[0]
            bbox = res.bboxes[0].tolist()
            output.append({
                'id': getattr(res, 'track_id', hash(bbox)),  # 兼容 MOT
                'keypoints': torch.cat([kpts, scores.unsqueeze(1)], dim=1).cpu().numpy(),
                'bbox': bbox,
                'score': res.scores.item() if hasattr(res, 'scores') else 1.0
            })
        return output
