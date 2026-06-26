# Zen Voyager

Image-to-video generation with explicit camera control.

> **Status: work in progress.** Built on a permissive, commercially-licensed foundation (no non-commercial or restricted-weight dependencies). The Zen model is not yet trained/integrated here — this provides the licensing, attribution, and integration scaffold.

## Foundation
Built on **Wan2.2-I2V** (Alibaba, Apache-2.0) with **MotionCtrl** (TencentARC, Apache-2.0) for camera-pose control — both permissive and commercial-friendly.

## License
Apache License — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright 2025-2026 Zen Authors (https://zenlm.org).

## Roadmap
- [ ] Wire Wan2.2-I2V + MotionCtrl camera conditioning
- [ ] Train the camera-control adapter on cleared data
- [ ] Inference, evaluation, and release
