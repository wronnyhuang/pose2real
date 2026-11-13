set -e
export DEBIAN_FRONTEND=noninteractive
(command -v ffmpeg >/dev/null && command -v rsync >/dev/null) || (apt-get update -qq && apt-get install -y -qq ffmpeg rsync >/dev/null)
pip install -q --break-system-packages diffusers transformers peft lpips accelerate opencv-python-headless soundfile 2>&1 | tail -2
python -c "
from huggingface_hub import snapshot_download
snapshot_download('stabilityai/sd-turbo', allow_patterns=['*.json','*.txt','unet/diffusion_pytorch_model.fp16.safetensors','vae/diffusion_pytorch_model.fp16.safetensors','text_encoder/model.fp16.safetensors','tokenizer/*'])
snapshot_download('facebook/hubert-base-ls960', allow_patterns=['*.json','pytorch_model.bin'])
import lpips; lpips.LPIPS(net='vgg')
"
python -c "import torch,diffusers,peft;print(torch.__version__,diffusers.__version__,peft.__version__, torch.cuda.get_device_name())"
echo SETUP_OK
