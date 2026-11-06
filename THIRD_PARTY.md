# Third-party components and licences

Nothing listed here is distributed in this repository. The add-on downloads models from Hugging Face and installs Python packages into a separate environment on the user's request. Each item keeps its own licence. **The summaries below are not legal advice**; they record what the model cards and repositories said when this file was written (November 2026). Check the current terms yourself, especially for commercial use of the models or of meshes made with them.

## Models

| Component | Used for | Licence as stated by its publisher | Commercial use |
|---|---|---|---|
| [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) (variants 2, 2mini, 2mv; code and weights) | optional neural start mesh (method 2) | Tencent Hunyuan 3D 2.0 Community License Agreement (`LICENSE` in the repository, checked November 2026) | Allowed inside the Territory, with conditions. The licence does not apply in the **European Union, United Kingdom and South Korea**, and use there is unlicensed. Licensees with more than 1 million monthly active users must ask Tencent for a licence. Output or the model may not be used to improve any other AI model. Subject to the Acceptable Use Policy. Read the full text |
| [Depth Anything V2 Large (HF)](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf) | depth maps when baking | CC BY-NC 4.0 (the Small model is Apache-2.0, per the [project README](https://github.com/DepthAnything/Depth-Anything-V2)) | Not permitted (NonCommercial) |
| [Marigold normals v1.1](https://huggingface.co/prs-eth/marigold-normals-v1-1) | normal maps when baking | Weights: Open RAIL++-M (`LICENSE-MODEL.txt` in the [Marigold repository](https://github.com/prs-eth/marigold)). Marigold code: Apache-2.0 | Allowed, with use-based restrictions that must be passed on to downstream users; read the licence |

Whether the licence of a model also covers the meshes produced with it depends on the licence text. This repository does not claim either way.

## Python packages installed on demand

Licences listed from general knowledge of these projects; verify before relying on them.

| Package | Licence |
|---|---|
| PyTorch, torchvision | BSD-style |
| diffusers, transformers, accelerate | Apache-2.0 |
| numpy, scipy, scikit-image | BSD-3-Clause |
| Pillow | HPND |
| trimesh | MIT |
| OpenCV (headless) | Apache-2.0 |
| einops | MIT |
| omegaconf | BSD-3-Clause |
| pymeshlab | GPL-3.0 |

## This repository

The add-on's own code: GPL-3.0-or-later (see `LICENSE`).
