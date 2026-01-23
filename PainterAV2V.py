import torch
import nodes
import node_helpers
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import comfy.clip_vision
from comfy_api.latest import io


def linear_interpolation(features, input_fps, output_fps, output_len=None):
    """
    features: shape=[num_layers, T, dim]
    input_fps: fps for audio encoder output (usually 50 for wav2vec2)
    output_fps: target video fps (user specified)
    """
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = torch.nn.functional.interpolate(
        features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


from comfy.ldm.wan.model_multitalk import InfiniteTalkOuterSampleWrapper, MultiTalkCrossAttnPatch, project_audio_features
import comfy.patcher_extension


class PainterAV2V(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PainterAV2V",
            category="conditioning/video_models",
            inputs=[
                io.Model.Input("model"),
                io.ModelPatch.Input("model_patch"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=832, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("height", default=480, min=16, max=nodes.MAX_RESOLUTION, step=16),
                io.Int.Input("length", default=81, min=1, max=nodes.MAX_RESOLUTION, step=4),
                io.Float.Input("fps", default=25.0, min=1.0, max=60.0, step=0.1, tooltip="Video frame rate for lip sync alignment"),
                io.AudioEncoderOutput.Input("audio_encoder"),
                io.Image.Input("video"),
                io.Mask.Input("mask", optional=True),
                io.Image.Input("start_image", optional=True),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Float.Input("audio_scale", default=1.0, min=-10.0, max=10.0, step=0.01),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def execute(cls, model, model_patch, positive, negative, vae, width, height, length, fps, audio_encoder, video, mask=None, start_image=None, clip_vision_output=None, audio_scale=1.0) -> io.NodeOutput:
        # VAE Encode video (VAEEncode functionality)
        latent_video = vae.encode(video)
        out_latent = {"samples": latent_video}
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
            out_latent["noise_mask"] = mask

        # Process start_image for conditioning (from WanInfiniteTalkToVideo)
        concat_latent_image = None
        if start_image is not None:
            start_image_proc = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.ones((length, height, width, start_image_proc.shape[-1]), device=start_image_proc.device, dtype=start_image_proc.dtype) * 0.5
            image[:start_image_proc.shape[0]] = start_image_proc

            concat_latent_image = vae.encode(image[:, :, :, :3])
            concat_mask = torch.ones((1, 1, ((length - 1) // 4) + 1, concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image_proc.device, dtype=start_image_proc.dtype)
            concat_mask[:, :, :((start_image_proc.shape[0] - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": concat_mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        # Process audio with configurable fps (from WanInfiniteTalkToVideo, single speaker only)
        model_patched = model.clone()

        all_layers = audio_encoder["encoded_audio_all_layers"]
        encoded_audio = torch.stack(all_layers, dim=0).squeeze(1)[1:]  # shape: [num_layers, T, 512]
        # Use user-specified fps instead of hardcoded 25
        encoded_audio = linear_interpolation(encoded_audio, input_fps=50, output_fps=fps).movedim(0, 1)  # shape: [T, num_layers, 512]

        # Single speaker mode: no two-speaker support, no previous frames
        encoded_audio_list = [encoded_audio]
        audio_start = 0
        audio_end = length

        # Prepare motion frames latent for outer sample wrapper
        # Use first frame of concat_latent_image if available, else zeros
        if concat_latent_image is not None:
            motion_frames_latent = concat_latent_image[:, :, :1]
        else:
            motion_frames_latent = torch.zeros([1, 16, 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())

        # Project audio features
        audio_embed = project_audio_features(model_patch.model.audio_proj, encoded_audio_list, audio_start, audio_end).to(model_patched.model_dtype())
        model_patched.model_options["transformer_options"]["audio_embeds"] = audio_embed

        # Add outer sample wrapper and patches (is_extend=False since no previous_frames)
        model_patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "infinite_talk_outer_sample",
            InfiniteTalkOuterSampleWrapper(
                motion_frames_latent,
                model_patch,
                is_extend=False,
            ))
        model_patched.set_model_patch(MultiTalkCrossAttnPatch(model_patch, audio_scale), "attn2_patch")

        return io.NodeOutput(model_patched, positive, negative, out_latent)
