# Commercial Rebase — Zen Voyager (image-to-video, camera control)

## Current base (RESTRICTED)
Tencent HunyuanWorld-Voyager, under the TENCENT HUNYUANWORLD-VOYAGER COMMUNITY
LICENSE (commercial use capped at 1M MAU; territorial/use restrictions).

## Target base (commercial-clean, all Apache-2.0)
- **Wan2.2-I2V** (https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) — Apache-2.0,
  unconditionally commercial image-to-video backbone.
- **MotionCtrl** (https://github.com/TencentARC/MotionCtrl) — Apache-2.0,
  plug-and-play camera-pose control. (Avoid CameraCtrl — academic-only.)

## Steps
1. Replace `voyager/` model with Wan2.2-I2V + MotionCtrl camera conditioning.
2. Retrain camera-control adapter on cleared data.
3. Publish weights to HF `zenlm/zen-voyager` under Apache-2.0.
4. Swap LICENSE -> Apache-2.0 + Wan/MotionCtrl attribution in NOTICE.
