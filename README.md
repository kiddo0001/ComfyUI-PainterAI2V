# ComfyUI-PainterAI2V

---
<img width="1011" height="894" alt="image" src="https://github.com/user-attachments/assets/0345c75c-63b6-4168-bc9b-a3834ce9dc99" />

## 节点特点

**专为 Wan2.2 双模型工作流优化的 InfiniteTalk 对口型节点，支持首尾帧精确控制**

- **帧率同步控制**：新增 `video_fps` 参数，可自定义设置视频帧率（1-120fps），音频口型自动匹配该帧率，完美解决原生硬编码 25fps 导致的音画不同步问题
- **Wan2.2 双模型架构**：同时支持高噪模型（0-2步）和低噪模型（2-4步）并行打补丁，保持与官方工作流完全一致
- **三模式首帧控制**：
  - **仅首帧**：传入 `start_image`，视频从首帧开始生成并全程对口型
  - **仅尾帧**：传入 `end_image`，视频最终定格指定画面并对口型
  - **首尾帧**：同时传入 `start_image` 和 `end_image`，视频从首帧自然过渡到尾帧，全程精准对口型
- **提示词动作运镜控制**：继承 Wan2.2 强大的提示词理解能力，可通过文本精确控制人物动作、相机运镜和场景变化
- **简化接口设计**：输入输出端口重命名为 `high_model`/`low_model`/`audio_encoder`，节点面板更清爽直观
- **全功能保留**：支持单人/双人对口型、motion context、previous frames 延续生成等所有原生功能

---<img width="2706" height="1375" alt="image" src="https://github.com/user-attachments/assets/6d4eea34-0de7-448f-bd32-d5cd2b4f7e30" />


## Node Features

**InfiniteTalk lip-sync node optimized for Wan2.2 dual-model workflow with first/last frame precision control**

- **FPS Synchronization Control**: New `video_fps` parameter allows custom video frame rate settings (1-120fps), with audio lip-sync automatically matching the specified rate, completely solving audio-visual desynchronization caused by the original hard-coded 25fps
- **Wan2.2 Dual-Model Architecture**: Simultaneously patches both high-noise (0-2 steps) and low-noise (2-4 steps) models, maintaining full compatibility with official workflows
- **Three First/Last Frame Modes**:
  - **First frame only**: Pass `start_image` to generate video from first frame with continuous lip-sync
  - **Last frame only**: Pass `end_image` to end video at specified frame with lip-sync
  - **First & Last frames**: Pass both `start_image` and `end_image` for natural transition from start to end with precise lip-sync throughout
- **Prompt-Controlled Motion & Camera**: Inherits Wan2.2's powerful prompt comprehension, enabling precise control of character movements, camera operations, and scene changes through text prompts
- **Simplified Interface**: Input/output ports renamed to `high_model`/`low_model`/`audio_encoder` for cleaner, more intuitive node panel
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


## 核心参数 / Core Parameters

| 参数 | 说明 | Parameter | Description |
|------|------|-----------|-------------|
| `video_fps` | 视频输出帧率，音频口型将自动匹配此速率 | Video output frame rate, audio lip-sync will automatically match this rate |
| `motion_frame` | 运动上下文帧数，用于延续生成 | Number of motion context frames for continuation |
| `audio_scale` | 口型强度系数，控制音频影响程度 | Lip-sync intensity coefficient, controls audio influence level |
| `mode` | 单/双说话人模式切换 | Single/dual speaker mode toggle |
| `start_image` | 首帧图像，视频从此帧开始 | First frame image, video starts from this frame |
| `end_image` | 尾帧图像，视频最终定格此画面 | Last frame image, video ends at this frame |

---

## 技术亮点 / Technical Highlights

- **智能帧率换算**：内部采用 50fps→`video_fps` 的动态插值，确保任意帧率下口型精准同步
- **双模型同步补丁**：同时对高/低噪模型应用 InfiniteTalk 补丁，避免采样阶段效果丢失
- **零漂移设计**：运动帧 latent 与音频特征在时序上严格对齐，保证长视频生成稳定性
- **显存友好**：支持逐批处理，大分辨率下可通过拆分降低显存占用
- **首帧尾帧 mask 保护**：首尾帧区域自动设置 mask=0，确保关键帧像素不被破坏

---

- **Intelligent Frame Rate Conversion**: Internal dynamic interpolation from 50fps→`video_fps` ensures precise lip-sync at any frame rate
- **Dual-Model Synchronous Patching**: Simultaneously applies InfiniteTalk patches to both high/low noise models, preventing effect loss during sampling stages
- **Zero-Drift Design**: Motion frame latents and audio features are strictly temporally aligned, ensuring stability for long video generation
- **Memory-Friendly**: Supports batch-by-batch processing, allows splitting to reduce VRAM usage at high resolutions
- **First/Last Frame Mask Protection**: Automatic mask=0 for first/last frame areas, ensuring keyframe pixels remain intact
