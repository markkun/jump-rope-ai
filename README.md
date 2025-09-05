# 🏃‍♂️ 跳绳 AI 分析系统（Jump Rope AI）

> 基于 MMPose + ST-GCN + 神经评分网络的高性能跳绳动作识别、计数与评分系统  
> 支持多人检测、实时分析、ONNX 导出与 Web 可视化

## 📋 功能特性

✅ **多人跳绳识别**：基于 YOLOX + RTMPose + ByteTrack 实现多人姿态跟踪  
✅ **高精度计数**：融合髋部、脚踝与节奏特征的智能计数算法  
✅ **动作评分系统**：神经网络模型自动评分（0-100）  
✅ **端到端流水线**：`JumpRopePipeline` 统一推理流程  
✅ **ONNX 支持**：支持 ST-GCN、ScoringNet、RTMPose 模型导出  
✅ **Web API + 前端**：Flask 接口 + React 可视化界面  
✅ **Docker 部署**：一键容器化部署，生产就绪

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/yourname/jump-rope-ai.git
cd jump-rope-ai
```
### 2. 安装依赖
```bash
pip install -r requirements.txt
```
### 3. 下载预训练模型
```bash
# 使用 mim 下载 MMPose 和 MMDet 模型
mim download mmdet configs/yolox/yolox_l_8x8_300e_coco.py --dest checkpoints/
mim download mmpose configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py --dest checkpoints/

# 下载 ST-GCN 模型（示例）
wget -O models/stgcn_best.pth https://example.com/stgcn_jump.pth
```

### 4. 实时摄像头分析
```bash
python app/real_time.py
```
## 📂 项目结构

jump-rope-ai/
├── src/                     # 核心模块
│   ├── optimized_pose_extractor.py    # 多人姿态提取
│   ├── optimized_counter.py           # 多特征计数
│   ├── scorer.py                      # 评分神经网络
│   ├── pipeline.py                    # 端到端推理流水线
│   └── ...
├── app/                     # 应用层
│   ├── real_time.py         # 实时摄像头分析
│   └── web_api.py           # Flask API 服务
├── scripts/                 # 工具脚本
│   ├── generate_scoring_data.py       # 评分数据生成
│   └── export_onnx_full.py            # ONNX 全流程导出
├── models/                  # 模型权重
│   ├── stgcn_best.pth
│   └── scoring_net.pth
├── data/                    # 训练数据
│   ├── scoring/features.npy
│   └── scoring/labels.npy
├── frontend/                # React 前端
├── config.yaml              # 配置文件
├── requirements.txt         # 依赖
├── Dockerfile               # 容器化
└── README.md
## 🛠️ 功能使用
### 1. 训练评分模型
```bash
# 生成训练数据（需准备视频和标签）
python scripts/generate_scoring_data.py

# 训练评分模型（示例）
python scripts/train_scoring.py
```

### 2. 导出 ONNX 模型

```bash
python scripts/export_onnx_full.py
```
输出：

models/stgcn.onnx
models/scoring_net.onnx
models/rtmpose.onnx（需使用 mim export）

### 3. 启动 Web API Bash 深色版本  
```bash
python app/web_api.py  访问 http://localhost:5000 查看文档。
```

### 4. 运行前端界面 Bash 深色版本  
```bash
cd frontend
npm install
npm start
```

## 🤝 贡献
欢迎提交 Issue 或 Pull Request！

## Fork 项目
创建特性分支 (git checkout -b feature/xxx)
提交更改 (git commit -m 'Add xxx')
推送到分支 (git push origin feature/xxx)
打开 Pull Request
## 📄 许可
本项目基于 MIT License 开源。

## 💌 鸣谢
MMPose
MMDetection
MMEngine
PyTorch
