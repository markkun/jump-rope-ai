FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app
COPY . .

RUN pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

# 预下载模型
RUN wget -O rtmpose.pth https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_120e-256x192-f1d8ece0_20230126.pth

CMD ["python", "app/web_api.py"]
