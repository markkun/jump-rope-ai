# src/export_onnx.py
from src.stgcn_model import STGCN
import torch

model = STGCN(num_classes=2)
model.load_state_dict(torch.load('models/checkpoints/stgcn_best.pth'))
model.eval()

x = torch.randn(1, 300, 17, 3)
torch.onnx.export(
    model, x, "models/onnx/stgcn_jump_rope.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0:"batch", 1:"time"}},
    opset_version=13
)
print("ONNX 模型已导出")
