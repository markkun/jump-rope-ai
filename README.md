# 🏃‍♂️ Jump Rope AI：基于 RTMPose + ST-GCN 的跳绳动作识别与计数系统

使用 **RTMPose 提取骨骼关键点 + ST-GCN 动作识别 + 峰值检测计数**，实现端到端跳绳检测与计数。

![](demo.gif) <!-- 可添加演示图 -->

## 🔧 功能

- ✅ 高精度骨骼提取（RTMPose）
- ✅ 跳绳动作分类（跳绳 vs 非跳绳）
- ✅ 自动跳绳计数（基于髋部运动）
- ✅ 数据增强与模型训练
- ✅ ONNX 导出，支持移动端部署
- ✅ 实时摄像头推理

## 📦 安装依赖

```bash
pip install -r requirements.txt
