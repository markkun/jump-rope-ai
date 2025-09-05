# scripts/export_onnx_full.py
"""
ONNX 全流程导出脚本
"""
import torch
import torch.onnx
from src.optimized_pose_extractor import OptimizedPoseExtractor
from src.stgcn_model import STGCN
from src.scorer import ScoringNet
import onnx
import onnxruntime as ort

def export_rtmpose_onnx(ckpt_path, output_path="models/rtmpose.onnx"):
    """导出 RTMPose（需通过 MMPose 导出）"""
    # 注意：MMPose 支持直接导出 ONNX
    print("💡 提示：使用 MMPose 命令导出 RTMPose")
    print("mim export mmpose rtmo-s_8xb32-80e_body8-halpe26-384x288.pytorch --format onnx")
    # 实际部署建议使用 MMPose 官方导出工具

def export_stgcn_onnx(model, output_path="models/stgcn.onnx", seq_len=300):
    model.eval()
    dummy_input = torch.randn(1, seq_len, 17, 3)  # [B, T, V, C]
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'seq_len'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ ST-GCN 已导出至 {output_path}")

    # 验证 ONNX 模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(output_path)
    outputs = ort_session.run(None, {'input': dummy_input.numpy()})
    print("✅ ST-GCN ONNX 验证通过，输出形状:", outputs[0].shape)

def export_scoring_net_onnx(model, output_path="models/scoring_net.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 6)  # 特征维度
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        input_names=['features'],
        output_names=['score'],
        dynamic_axes={'features': {0: 'batch'}}
    )
    print(f"✅ ScoringNet 已导出至 {output_path}")

    # 验证
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(output_path)
    outputs = ort_session.run(None, {'features': dummy_input.numpy()})
    print("✅ ScoringNet ONNX 验证通过，输出:", outputs[0])

if __name__ == "__main__":
    # 导出 ST-GCN
    stgcn_model = STGCN(num_classes=2)
    stgcn_model.load_state_dict(torch.load("models/stgcn_best.pth"))
    export_stgcn_onnx(stgcn_model)

    # 导出 ScoringNet
    scorer_model = ScoringNet()
    try:
        scorer_model.load_state_dict(torch.load("models/scoring_net.pth"))
    except:
        print("⚠️  使用随机权重导出（首次）")
    export_scoring_net_onnx(scorer_model)

    # RTMPose 提示
    export_rtmpose_onnx(None)
