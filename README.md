# ComfyUI-PainterAI2V

---
<img width="1067" height="928" alt="image" src="https://github.com/user-attachments/assets/ec7df47f-3013-4cb9-ae09-30f191713c08" />


## 节点特点

**专为 Wan2.2 双模型工作流优化的 InfiniteTalk 对口型节点**

- **帧率同步控制**：新增 `video_fps` 参数，可自定义设置视频帧率，音频口型自动匹配该帧率，完美解决原生硬编码 25fps 导致的音画不同步问题
- **Wan2.2 双模型架构**：同时支持高噪模型（0-2步）和低噪模型（2-4步）并行打补丁，保持与官方工作流完全一致
- **提示词动作控制**：继承 Wan2.2 强大的提示词理解能力，可通过文本精确控制人物动作、运镜和场景变化
- **全功能保留**：支持单人/双人对口型、motion context、previous frames 延续生成等所有原生功能
<img width="2153" height="1211" alt="image" src="https://github.com/user-attachments/assets/c2f59da9-9a1f-491f-b544-8f467f15884c" />

---

## Node Features

**InfiniteTalk lip-sync node optimized for Wan2.2 dual-model workflow**

- **FPS Synchronization Control**: New `video_fps` parameter allows custom video frame rate settings, with audio lip-sync automatically matching the specified rate, completely solving the audio-visual desynchronization caused by the original hard-coded 25fps
- **Wan2.2 Dual-Model Architecture**: Simultaneously patches both high-noise (0-2 steps) and low-noise (2-4 steps) models, maintaining full compatibility with official workflows
- **Prompt-Controlled Motion**: Inherits Wan2.2's powerful prompt comprehension capability, enabling precise control of character movements, camera operations, and scene changes through text prompts
- **Full Feature Retention**: Supports all native features including single/dual speaker lip-sync, motion context, and previous frames continuation

---

## 安装 / Installation

1. 将节点包放入 `ComfyUI/custom_nodes/` 目录
2. 重启 ComfyUI
3. 节点位于 `conditioning/video_models` 分类下

---

1. Place the node package in `ComfyUI/custom_nodes/` directory
2. Restart ComfyUI
3. Node is located in `conditioning/video_models` category

---

## 工作流说明 / Workflow Instructions

**标准使用流程：**
1. 加载 Wan2.2 高噪模型 → 连接 `high_model`
2. 加载 Wan2.2 低噪模型 → 连接 `low_model`
3. 加载 InfiniteTalk 补丁模型 → 连接 `model_patch`
4. 设置 `video_fps` 与视频合成节点帧率一致
5. 音频编码器输出 → 连接 `audio_encoder`
6. 提示词正向/负向 → 连接 `positive`/`negative`
7. 输出连接到 KSamplerAdvanced 采样器

**Standard workflow:**
1. Load Wan2.2 high-noise model → connect to `high_model`
2. Load Wan2.2 low-noise model → connect to `low_model`
3. Load InfiniteTalk patch model → connect to `model_patch`
4. Set `video_fps` to match video combine node frame rate
5. Audio encoder output → connect to `audio_encoder`
6. Positive/negative prompts → connect to `positive`/`negative`
7. Output connects to KSamplerAdvanced sampler

---

## 核心参数 / Core Parameters

| 参数 | 说明 | Parameter | Description |
|------|------|-----------|-------------|
| `video_fps` | 视频输出帧率，音频口型将自动匹配此速率 | Video output frame rate, audio lip-sync will automatically match this rate |
| `motion_frame` | 运动上下文帧数，用于延续生成 | Number of motion context frames for continuation |
| `audio_scale` | 口型强度系数，控制音频影响程度 | Lip-sync intensity coefficient, controls audio influence level |
| `mode` | 单/双说话人模式切换 | Single/dual speaker mode toggle |

---

## 技术亮点 / Technical Highlights

- **智能帧率换算**：内部采用 50fps→`video_fps` 的动态插值，确保任意帧率下口型精准同步
- **双模型同步补丁**：同时对高/低噪模型应用 InfiniteTalk 补丁，避免采样阶段效果丢失
- **零漂移设计**：运动帧 latent 与音频特征在时序上严格对齐，保证长视频生成稳定性
- **显存友好**：支持逐批处理，大分辨率下可通过拆分降低显存占用

---

- **Intelligent Frame Rate Conversion**: Internal dynamic interpolation from 50fps→`video_fps` ensures precise lip-sync at any frame rate
- **Dual-Model Synchronous Patching**: Simultaneously applies InfiniteTalk patches to both high/low noise models, preventing effect loss during sampling stages
- **Zero-Drift Design**: Motion frame latents and audio features are strictly temporally aligned, ensuring stability for long video generation
- **Memory-Friendly**: Supports batch-by-batch processing, allows splitting to reduce VRAM usage at high resolutions
