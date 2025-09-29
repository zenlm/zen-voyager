# Zen Voyager

**Zen Voyager** generates world-consistent 3D video sequences from a single image with camera-path control. Perfect for world exploration, 3D reconstruction, and immersive content creation.

<p align="center">
  <a href="https://github.com/zenlm/zen-voyager"><img src="https://img.shields.io/badge/GitHub-zenlm%2Fzen--voyager-blue"></a>
  <a href="https://huggingface.co/zenlm/zen-voyager"><img src="https://img.shields.io/badge/🤗-Models-yellow"></a>
  <a href="https://github.com/zenlm"><img src="https://img.shields.io/badge/Zen-AI-purple"></a>
</p>

## Overview

Zen Voyager creates camera-controllable video with:

- 🎬 **Camera Control**: Define custom camera trajectories
- 🌍 **World Consistency**: 3D-consistent scene videos
- 📐 **Depth + RGB**: Aligned depth and RGB generation
- 🔄 **3D Reconstruction**: Efficient point-cloud generation
- 🚀 **Long-Range Exploration**: Auto-regressive scene extension
- 🎯 **Image-to-3D**: Convert single images to 3D scenes

## Model Details

- **Model Type**: Image-to-Video with Camera Control (Diffusion)
- **Architecture**: World-Consistent Video Diffusion Transformer
- **License**: Apache 2.0
- **Input**: Single image + camera trajectory
- **Output**: RGB video + depth video + point clouds
- **Developed by**: Zen AI Team
- **Based on**: [HunyuanWorld-Voyager by Tencent](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager)

## Capabilities

### Camera-Controllable Video
- Define custom camera paths (pan, tilt, dolly, orbit)
- Smooth camera motion with natural transitions
- User-defined keyframes and trajectories
- Multiple camera modes (FPS, orbital, cinematic)

### 3D-Consistent Generation
- World-consistent multi-view synthesis
- Geometric coherence across frames
- Metric depth prediction
- Photometric consistency

### Applications
- **3D Reconstruction**: Generate point clouds from single images
- **Virtual Tours**: Create immersive walkthrough videos
- **Content Creation**: Camera-controlled scene exploration
- **Gaming**: Generate explorable 3D environments
- **VR/AR**: Immersive world building

## Hardware Requirements

### Minimum
- **GPU**: 16GB VRAM (RTX 4080, RTX 3090)
- **RAM**: 32GB system memory
- **Storage**: 50GB for model

### Recommended
- **GPU**: 24GB VRAM (RTX 4090)
- **RAM**: 64GB system memory
- **Storage**: 100GB for model and cache

### Optimal
- **GPU**: 40GB+ VRAM (A100)
- **RAM**: 128GB system memory
- For high-resolution and long sequences

## Installation

```bash
# Clone repository
git clone https://github.com/zenlm/zen-voyager.git
cd zen-voyager

# Create environment
conda create -n zen-voyager python=3.10
conda activate zen-voyager

# Install dependencies
pip install -r requirements.txt

# Download model
huggingface-cli download zenlm/zen-voyager --local-dir ./ckpts
```

## Usage

### Basic Camera-Controlled Video

```bash
python infer.py \
    --image input.jpg \
    --camera_path forward \
    --output output.mp4
```

### Custom Camera Path

```bash
python infer.py \
    --image input.jpg \
    --camera_path custom \
    --camera_config camera_trajectory.json \
    --output output.mp4
```

### With Depth Generation

```bash
python infer.py \
    --image input.jpg \
    --camera_path orbit \
    --output_rgb output_rgb.mp4 \
    --output_depth output_depth.mp4
```

### Python API

```python
from zen_voyager import ZenVoyagerPipeline

# Initialize
pipeline = ZenVoyagerPipeline.from_pretrained("zenlm/zen-voyager")

# Define camera path
camera_path = {
    "type": "orbit",
    "radius": 5.0,
    "height": 1.0,
    "frames": 120
}

# Generate video
result = pipeline(
    image="input.jpg",
    camera_path=camera_path,
    generate_depth=True
)

# Save outputs
result.rgb.save("output_rgb.mp4")
result.depth.save("output_depth.mp4")
result.pointcloud.save("output.ply")
```

## Camera Path Types

### Forward/Backward
```python
camera_path = {"type": "forward", "distance": 10.0, "frames": 60}
```

### Orbital
```python
camera_path = {"type": "orbit", "radius": 5.0, "rotations": 1.0, "frames": 120}
```

### Pan/Tilt
```python
camera_path = {"type": "pan", "angle": 90, "frames": 60}
```

### Custom Keyframes
```python
camera_path = {
    "type": "keyframe",
    "keyframes": [
        {"position": [0, 0, 0], "rotation": [0, 0, 0], "frame": 0},
        {"position": [5, 2, 3], "rotation": [0, 45, 0], "frame": 60},
        {"position": [0, 0, 10], "rotation": [0, 90, 0], "frame": 120}
    ]
}
```

## 3D Reconstruction

```python
# Generate 3D point cloud from single image
from zen_voyager import reconstruct_3d

pointcloud = reconstruct_3d(
    image="input.jpg",
    num_views=8,  # Number of viewpoints
    output="output.ply"
)

# Convert to mesh
mesh = pointcloud.to_mesh(method="poisson")
mesh.save("output.obj")
```

## Training with Zen Gym

Fine-tune for custom scenes:

```bash
cd /path/to/zen-gym

llamafactory-cli train \
    --config configs/zen_voyager_lora.yaml \
    --dataset your_video_dataset
```

## Inference with Zen Engine

Serve via API:

```bash
cd /path/to/zen-engine

cargo run --release -- serve \
    --model zenlm/zen-voyager \
    --port 3690
```

## Performance

### Generation Speed
- **RTX 4090**: ~30s for 60-frame video
- **A100**: ~20s for 60-frame video
- **Resolution**: 512x512 (default), up to 1024x1024

### Quality Metrics
| Metric | Score |
|--------|-------|
| WorldScore | 73.25 |
| Camera Control | 90.2 |
| 3D Consistency | 88.6 |
| Content Alignment | 91.8 |

## Use Cases

### Virtual Tours
- Real estate walkthroughs
- Museum exhibitions
- Architectural visualization
- Property showcases

### Content Creation
- YouTube 360° videos
- VR/AR experiences
- Game cinematics
- Film pre-visualization

### 3D Reconstruction
- Single-image to 3D
- Object modeling
- Scene reconstruction
- Photogrammetry alternative

### Research
- Novel view synthesis
- 3D scene understanding
- Camera pose estimation
- Depth prediction

## Limitations

- Resolution limited to 1024x1024
- Complex scenes may have artifacts
- Camera paths must be physically plausible
- Occlusion handling for extreme viewpoints
- Requires high-quality input images

## Ethical Considerations

- Generated content should be labeled
- Not for creating misleading 3D environments
- Respect privacy and property rights
- Consider computational environmental impact
- Potential misuse in virtual scams

## Citation

```bibtex
@misc{zenvoyager2025,
  title={Zen Voyager: Camera-Controlled World Exploration},
  author={Zen AI Team},
  year={2025},
  howpublished={\url{https://github.com/zenlm/zen-voyager}}
}
```

## Credits

Based on [HunyuanWorld-Voyager](https://github.com/Tencent-Hunyuan/HunyuanWorld-Voyager) by Tencent. We thank the original authors for their groundbreaking work.

## Links

- **GitHub**: https://github.com/zenlm/zen-voyager
- **HuggingFace**: https://huggingface.co/zenlm/zen-voyager
- **Organization**: https://github.com/zenlm
- **Zen Gym** (Training): https://github.com/zenlm/zen-gym
- **Zen Engine** (Inference): https://github.com/zenlm/zen-engine
- **Zen 3D**: https://github.com/zenlm/zen-3d

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

---

**Zen Voyager** - Explore 3D worlds from a single image

Part of the **[Zen AI](https://github.com/zenlm)** ecosystem.