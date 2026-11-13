#!/bin/bash
# Base models: stabilityai/sd-turbo, facebook/hubert-base-ls960 (both HF). SyncNet (eval) from joonson/syncnet_python.
set -e; cd "$(dirname "$0")/.."
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python - <<'P'
from huggingface_hub import snapshot_download
snapshot_download('stabilityai/sd-turbo', allow_patterns=['*.json','*.txt','unet/diffusion_pytorch_model.fp16.safetensors','vae/diffusion_pytorch_model.fp16.safetensors','text_encoder/model.fp16.safetensors','tokenizer/*'])
snapshot_download('facebook/hubert-base-ls960', allow_patterns=['*.json','pytorch_model.bin'])
import lpips; lpips.LPIPS(net='vgg')
P
mkdir -p common/models eval/third_party/weights; B=https://storage.googleapis.com/mediapipe-models
curl -L -o common/models/face_landmarker.task $B/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
curl -L -o common/models/pose_landmarker_heavy.task $B/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
[ -d eval/third_party/syncnet_python/.git ] || echo "SyncNet: git clone https://github.com/joonson/syncnet_python into eval/third_party/syncnet_python, then run its download_model.sh and copy data/syncnet_v2.model + detectors/s3fd/weights/sfd_face.pth into eval/third_party/weights/ (eval/README.md lists the sha256 prefixes)"
echo SETUP_OK
