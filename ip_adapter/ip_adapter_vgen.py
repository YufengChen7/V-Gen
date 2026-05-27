import math
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from diffusers.pipelines.controlnet import MultiControlNetModel
from peft import LoraConfig
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from .utils import is_torch2_available, get_generator
if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from .attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler


def load_lora_model(unet, device, lora_rank, lora_alpha):
    for param in unet.parameters():
        param.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    unet.add_adapter(unet_lora_config)
    return unet


def _strip_module_prefix(state: Dict[str, Any]) -> Dict[str, Any]:
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}


def _unwrap_state_dict(state: Any) -> Any:
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict):
        state = _strip_module_prefix(state)
    return state


def _normalize_vgen_state(state: Any) -> Dict[str, Dict[str, torch.Tensor]]:
    state = _unwrap_state_dict(state)
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint object type: {type(state)}")

    if "image_proj" in state and "ip_adapter" in state:
        return {
            "image_proj": state.get("image_proj", {}),
            "ip_adapter": state.get("ip_adapter", {}),
            "unet": state.get("unet", {}),
        }

    if "image_proj_model" in state or "adapter_modules" in state or "ip_adapter_model" in state:
        image_proj_sd = state.get("image_proj_model", {})
        ip_sd = state.get("adapter_modules", state.get("ip_adapter_model", {}))
        unet_sd = state.get("unet", {})
        if isinstance(image_proj_sd, dict) and isinstance(ip_sd, dict):
            return {
                "image_proj": image_proj_sd,
                "ip_adapter": ip_sd,
                "unet": unet_sd if isinstance(unet_sd, dict) else {},
            }

    image_proj_sd = {}
    ip_sd = {}
    unet_sd = {}
    for k, v in state.items():
        if k.startswith("image_proj_model."):
            image_proj_sd[k.replace("image_proj_model.", "", 1)] = v
        elif k.startswith("adapter_modules."):
            ip_sd[k.replace("adapter_modules.", "", 1)] = v
        elif k.startswith("ip_adapter_model."):
            ip_sd[k.replace("ip_adapter_model.", "", 1)] = v
        elif k.startswith("unet."):
            unet_sd[k.replace("unet.", "", 1)] = v

    if image_proj_sd or ip_sd or unet_sd:
        return {"image_proj": image_proj_sd, "ip_adapter": ip_sd, "unet": unet_sd}

    raise ValueError(
        "Unexpected Vgen checkpoint format. Expected one of: "
        "packed(image_proj/ip_adapter[/unet]), nested(image_proj_model/adapter_modules), or raw prefixed keys."
    )


def _normalize_attention_state(state: Any) -> Dict[str, torch.Tensor]:
    state = _unwrap_state_dict(state)
    if isinstance(state, dict) and "att" in state and isinstance(state["att"], dict):
        return _strip_module_prefix(state["att"])
    if isinstance(state, dict):
        return _strip_module_prefix(state)
    raise ValueError(f"Unexpected attention checkpoint object type: {type(state)}")


def _infer_lora_rank_from_unet_state(unet_state: Dict[str, torch.Tensor]) -> Optional[int]:
    for key, value in unet_state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if "lora_A" in key and key.endswith(".weight") and value.ndim == 2:
            return int(value.shape[0])
    return None


def _positive_int_or_none(value: Any) -> Optional[int]:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _zero_lora_weights(module: torch.nn.Module) -> int:
    zeroed = 0
    for name, param in module.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            with torch.no_grad():
                param.zero_()
            zeroed += 1
    return zeroed


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        b = embeds.shape[0]
        # clip_extra_context_tokens = self.proj(embeds).reshape(
        #     -1, self.clip_extra_context_tokens, self.cross_attention_dim
        # )
        clip_extra_context_tokens = self.proj(embeds).reshape(
            b, -1, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class SelfAttention(nn.Module):
    def __init__(self, in_channels, device):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1).to(device)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1).to(device)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1).to(device)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1).to(device)
        self.proj_out = nn.Linear(1280, 1024).to(device)

    def forward(self, x, mask=None):
        x = x.permute(0, 2, 1)
        batch_size, channels, h = x.size()
        height = int(math.sqrt(h))
        width = height
        x = x.view(batch_size, channels, width, height)
        batch_size, channels, height, width = x.size()
        # Compute query, key, and value tensors on the token grid.
        q = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height * width)
        v = self.value(x).view(batch_size, -1, height * width)

        attention_scores = torch.bmm(q, k)

        if mask is not None:
            mask = nn.functional.interpolate(mask, size=(height, width), mode='nearest')
            mask = mask.view(batch_size, 1, height * width)
            large_constant = 1e6
            attention_scores = attention_scores - (1 - mask) * large_constant

        attention_weights = self.softmax(attention_scores)

        out = torch.bmm(v, attention_weights.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # Residual attention output.
        out = self.gamma.to(x.device) * out + x
        out = out.view(batch_size, channels, height * width)
        out = out.permute(0, 2, 1)
        out = self.proj_out(out)

        return out


class Vgen:
    def __init__(
        self,
        sd_pipe,
        image_encoder_path,
        ip_ckpt,
        ip_ckpt_1,
        device,
        num_tokens=4,
        lora_rank=-1,
        lora_alpha=-1,
        strict_unet_load=False,
        fail_on_unet_mismatch=False,
        load_unet_weights=True,
    ):
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.ip_ckpt_1 = ip_ckpt_1
        self.num_tokens = num_tokens
        self.strict_unet_load = strict_unet_load
        self.fail_on_unet_mismatch = fail_on_unet_mismatch
        self.load_unet_weights = load_unet_weights
        self.load_report = {}
        self.attention_module = SelfAttention(1280, device)
        self.pipe = sd_pipe.to(self.device)

        # Preload and normalize checkpoint first so LoRA rank can be inferred when not provided.
        vgen_state = self._load_vgen_state(self.ip_ckpt)
        inferred_rank = _infer_lora_rank_from_unet_state(vgen_state.get("unet", {}))
        self.lora_rank = _positive_int_or_none(lora_rank) or inferred_rank or 8
        self.lora_alpha = _positive_int_or_none(lora_alpha) or self.lora_rank

        print(
            f"[load] LoRA config for inference: rank={self.lora_rank}, alpha={self.lora_alpha}, "
            f"strict_unet_load={self.strict_unet_load}, load_unet_weights={self.load_unet_weights}"
        )
        self.set_vgen()

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )

        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj()

        self.load_vgen(vgen_state)

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)

        return image_proj_model

    def _load_raw_checkpoint(self, ckpt_path: str):
        if os.path.splitext(ckpt_path)[-1] == ".safetensors":
            state = {}
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state[key] = f.get_tensor(key)
            return state
        return torch.load(ckpt_path, map_location="cpu")

    def _load_vgen_state(self, ckpt_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
        raw_state = self._load_raw_checkpoint(ckpt_path)
        return _normalize_vgen_state(raw_state)

    def _load_attention_state(self, ckpt_path: str) -> Dict[str, torch.Tensor]:
        raw_state = self._load_raw_checkpoint(ckpt_path)
        return _normalize_attention_state(raw_state)

    def set_vgen(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float16)
        unet.set_attn_processor(attn_procs)
        unet = load_lora_model(unet, self.device, self.lora_rank, self.lora_alpha)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))

    def load_vgen(self, state_dict=None):
        if state_dict is None:
            state_dict = self._load_vgen_state(self.ip_ckpt)

        self.load_report = {
            "image_proj_keys": len(state_dict.get("image_proj", {})),
            "ip_adapter_keys": len(state_dict.get("ip_adapter", {})),
            "unet_keys": len(state_dict.get("unet", {})),
            "unet_loaded": False,
            "unet_missing_keys": 0,
            "unet_unexpected_keys": 0,
        }

        if "image_proj" not in state_dict or "ip_adapter" not in state_dict:
            raise ValueError(
                "ip_ckpt format invalid: expected keys image_proj and ip_adapter after normalization."
            )

        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

        unet_state = state_dict.get("unet", {})
        if self.load_unet_weights and isinstance(unet_state, dict) and len(unet_state) > 0:
            conv_in_weight = unet_state.get("conv_in.weight")
            if conv_in_weight is not None:
                ckpt_conv_in_channels = conv_in_weight.shape[1]
                model_conv_in_channels = self.pipe.unet.conv_in.weight.shape[1]
                if ckpt_conv_in_channels != model_conv_in_channels:
                    msg = (
                        "skip UNet loading due to channel mismatch: "
                        f"ckpt conv_in={ckpt_conv_in_channels}, model conv_in={model_conv_in_channels}"
                    )
                    if self.fail_on_unet_mismatch:
                        raise RuntimeError(msg)
                    print(f"[warn] {msg}")
                    unet_state = {}

            if len(unet_state) > 0:
                try:
                    if self.strict_unet_load:
                        self.pipe.unet.load_state_dict(unet_state, strict=True)
                        missing_keys = []
                        unexpected_keys = []
                    else:
                        incompatible = self.pipe.unet.load_state_dict(unet_state, strict=False)
                        missing_keys = list(incompatible.missing_keys)
                        unexpected_keys = list(incompatible.unexpected_keys)
                    self.load_report["unet_loaded"] = True
                    self.load_report["unet_missing_keys"] = len(missing_keys)
                    self.load_report["unet_unexpected_keys"] = len(unexpected_keys)
                    if not self.strict_unet_load and (missing_keys or unexpected_keys):
                        print(
                            f"[warn] non-strict UNet load: missing={len(missing_keys)}, "
                            f"unexpected={len(unexpected_keys)}"
                        )
                except Exception as e:
                    msg = f"failed to load UNet weights from ip_ckpt ({e})"
                    if self.fail_on_unet_mismatch:
                        raise RuntimeError(msg) from e
                    print(f"[warn] {msg}; continue with base UNet.")
        else:
            if not self.load_unet_weights:
                print("[info] UNet loading disabled by configuration.")
            else:
                print("[warn] 'unet' key missing or empty in ip_ckpt; continue with base UNet.")

        if not self.load_report["unet_loaded"]:
            zeroed = _zero_lora_weights(self.pipe.unet)
            if zeroed > 0:
                print(f"[warn] UNet checkpoint not loaded; reset {zeroed} LoRA tensors to zero for safe fallback.")

        attention_state = self._load_attention_state(self.ip_ckpt_1)
        self.attention_module.load_state_dict(attention_state)
        self.load_report["attention_keys"] = len(attention_state)
        print(
            "[load] checkpoint summary: "
            f"image_proj={self.load_report['image_proj_keys']} keys, "
            f"ip_adapter={self.load_report['ip_adapter_keys']} keys, "
            f"unet={self.load_report['unet_keys']} keys, "
            f"unet_loaded={self.load_report['unet_loaded']}, "
            f"attention={self.load_report['attention_keys']} keys"
        )

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, mask_image_0=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
            outputs = self.image_encoder(clip_image.to(self.device, dtype=torch.float16))
            clip_image_embeds = outputs.image_embeds
            last_feature_layer_output = outputs.last_hidden_state
        else:
            raise ValueError(
                "get_image_embeds requires pil_image. The current attention module depends on CLIP last_hidden_state."
            )
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        if mask_image_0 is None:
            raise ValueError("get_image_embeds requires mask_image_0 for attention conditioning.")
        if isinstance(mask_image_0, Image.Image):
            mask_w, mask_h = mask_image_0.size
        elif isinstance(mask_image_0, list) and mask_image_0 and isinstance(mask_image_0[0], Image.Image):
            mask_w, mask_h = mask_image_0[0].size
            mask_image_0 = mask_image_0[0]
        else:
            raise TypeError("mask_image_0 must be a PIL image.")
        target_w = max(1, int(mask_w) // 8)
        target_h = max(1, int(mask_h) // 8)
        mask_image_0 = mask_image_0.resize((target_w, target_h), resample=Image.NEAREST)
        mask_image_0 = mask_image_0.convert('L')
        mask_image_0 = torch.tensor(np.array(mask_image_0), dtype=torch.float32)
        # Input mask is 8-bit grayscale (0..255); threshold at 127 for stable binarization.
        mask_image_0 = (mask_image_0 > 127.0).float().to(self.device)
        image_embeds = self.attention_module(last_feature_layer_output[:, :256, :].float(),
                                             mask_image_0.unsqueeze(0).unsqueeze(0))
        image_prompt_embeds = self.image_proj_model(image_embeds.half())
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(image_embeds).half())
        # uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds).half())
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale

    def encode_long_text(self,
            input_ids: torch.Tensor,  # Token IDs from the tokenizer.
            tokenizer: CLIPTokenizer,
            text_encoder: CLIPTextModel,
            max_length: int = 77,  # CLIP token limit per chunk.
            device: str = "cuda"
    ) -> torch.Tensor:
        """
        Encode tokenized long text by splitting it into CLIP-length chunks and
        averaging the chunk embeddings.

        Args:
            input_ids: Tokenized input IDs, shape [batch_size, seq_len] or [seq_len].
            tokenizer: CLIP tokenizer.
            text_encoder: CLIP text encoder.
            max_length: Maximum number of tokens per chunk.
            device: Device used for encoding.

        Returns:
            Text embeddings with shape [batch_size, 1, hidden_dim].
        """
        # Ensure input is a 2D tensor: [batch_size, seq_len].
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [1, seq_len]

        batch_size = input_ids.size(0)
        hidden_dim = text_encoder.config.hidden_size

        combined_embeddings = torch.zeros(batch_size, hidden_dim, device=device)

        for batch_idx in range(batch_size):
            # Current sample token IDs.
            current_input_ids = input_ids[batch_idx]  # [seq_len]

            # Split by CLIP's per-call token limit.
            chunks = [
                current_input_ids[i:i + max_length]
                for i in range(0, len(current_input_ids), max_length)
            ]

            # Encode each chunk and collect its pooled feature.
            embeddings = []
            for chunk in chunks:
                # Add batch dimension and pad to max_length.
                chunk_len = len(chunk)
                padding_len = max_length - chunk_len

                # Build text encoder input.
                chunk_input = {
                    "input_ids": torch.cat([
                        chunk.unsqueeze(0).to(device),  # [1, chunk_len]
                        torch.zeros(1, padding_len, dtype=torch.long, device=device)  # [1, padding_len]
                    ], dim=1),  # [1, max_length]

                    "attention_mask": torch.cat([
                        torch.ones(1, chunk_len, dtype=torch.long, device=device),  # [1, chunk_len]
                        torch.zeros(1, padding_len, dtype=torch.long, device=device)  # [1, padding_len]
                    ], dim=1)  # [1, max_length]
                }

                with torch.no_grad():
                    chunk_emb = text_encoder(**chunk_input).last_hidden_state  # [1, max_length, hidden_dim]
                    # Average only the non-padding token features.
                    embeddings.append(chunk_emb[:, :chunk_len, :].mean(dim=1))

            if embeddings:
                combined_embeddings[batch_idx] = torch.mean(torch.cat(embeddings, dim=0), dim=0)
            else:
                combined_embeddings[batch_idx] = torch.zeros(hidden_dim, device=device)

        return combined_embeddings.unsqueeze(1)
    # def encode_long_text(
    #         self,
    #         text: Union[str, List[str]],
    #         tokenizer: CLIPTokenizer,
    #         text_encoder: CLIPTextModel,
    #         max_length: int = 77,
    #         device: Optional[str] = None
    # ) -> torch.Tensor:
    #     """
    #     Encode long text by splitting into chunks and averaging embeddings
    #
    #     Args:
    #         text: Input text or list of texts
    #         tokenizer: CLIP tokenizer
    #         text_encoder: CLIP text encoder
    #         max_length: Maximum token length per chunk
    #         device: Device to use (defaults to self.device)
    #
    #     Returns:
    #         torch.Tensor: Text embeddings [batch_size, seq_len, hidden_dim]
    #     """
    #     device = device or self.device
    #
    #     # Tokenize input text
    #     if isinstance(text, str):
    #         text = [text]
    #
    #     # Tokenize without truncation
    #     inputs = tokenizer(
    #         text,
    #         padding=False,
    #         truncation=False,
    #         return_tensors="pt",
    #         max_length=None
    #     )
    #     input_ids = inputs.input_ids.to(device)
    #     attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None
    #
    #     batch_size, seq_len = input_ids.shape
    #
    #     # Calculate number of chunks needed
    #     num_chunks = (seq_len + max_length - 1) // max_length
    #
    #     # Initialize embeddings tensor
    #     embeddings = []
    #
    #     for i in range(num_chunks):
    #         start_idx = i * max_length
    #         end_idx = (i + 1) * max_length
    #
    #         # Get chunk
    #         chunk_input_ids = input_ids[:, start_idx:end_idx]
    #         chunk_attention_mask = attention_mask[:, start_idx:end_idx] if attention_mask is not None else None
    #
    #         # Pad if needed
    #         padding_len = max_length - chunk_input_ids.shape[1]
    #         if padding_len > 0:
    #             padding = torch.zeros(batch_size, padding_len, dtype=torch.long, device=device)
    #             chunk_input_ids = torch.cat([chunk_input_ids, padding], dim=1)
    #             if chunk_attention_mask is not None:
    #                 chunk_attention_mask = torch.cat([
    #                     chunk_attention_mask,
    #                     torch.zeros(batch_size, padding_len, dtype=torch.long, device=device)
    #                 ], dim=1)
    #
    #         # Encode chunk
    #         with torch.no_grad():
    #             outputs = text_encoder(
    #                 input_ids=chunk_input_ids,
    #                 attention_mask=chunk_attention_mask,
    #                 return_dict=True
    #             )
    #             chunk_embeddings = outputs.last_hidden_state
    #
    #             # Apply attention mask if available
    #             if chunk_attention_mask is not None:
    #                 chunk_embeddings = chunk_embeddings * chunk_attention_mask.unsqueeze(-1)
    #
    #             # Average over sequence length (excluding padding)
    #             if chunk_attention_mask is not None:
    #                 valid_lengths = chunk_attention_mask.sum(dim=1, keepdim=True)
    #                 chunk_embeddings = (chunk_embeddings.sum(dim=1) / valid_lengths.clamp(min=1))
    #             else:
    #                 chunk_embeddings = chunk_embeddings.mean(dim=1)
    #
    #             embeddings.append(chunk_embeddings)
    #
    #     # Combine chunk embeddings by averaging
    #     if embeddings:
    #         combined_embeddings = torch.stack(embeddings, dim=1)  # [batch_size, num_chunks, hidden_dim]
    #         combined_embeddings = combined_embeddings.mean(dim=1)  # [batch_size, hidden_dim]
    #     else:
    #         combined_embeddings = torch.zeros(batch_size, text_encoder.config.hidden_size, device=device)
    #
    #     return combined_embeddings.unsqueeze(1)  # Add sequence dimension [batch_size, 1, hidden_dim]

    def generate(
            self,
            pil_image=None,
            clip_image_embeds=None,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            guidance_scale=7.5,
            # guidance_scale=10,
            num_inference_steps=30,
            mask_image_0=None,
            **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        # print("prompt:", prompt)
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds, mask_image_0=mask_image_0,
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            # Encode text prompts with the same long-text chunking path.
            prompt_embeds_list = []
            for p in prompt:
                # Convert text to token IDs.
                inputs = self.pipe.tokenizer(
                    p,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inputs.input_ids.to(self.device)  # [1, seq_len]

                prompt_embed = self.encode_long_text(
                    input_ids=input_ids,
                    tokenizer=self.pipe.tokenizer,
                    text_encoder=self.pipe.text_encoder,
                    device=self.device
                )  # [1, 1, hidden_dim]

                prompt_embeds_list.append(prompt_embed)

            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)

            # Encode negative prompts with the same path.
            negative_prompt_embeds_list = []
            for p in negative_prompt:
                inputs = self.pipe.tokenizer(
                    p,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = inputs.input_ids.to(self.device)

                negative_prompt_embed = self.encode_long_text(
                    input_ids=input_ids,
                    tokenizer=self.pipe.tokenizer,
                    text_encoder=self.pipe.text_encoder,
                    device=self.device
                )

                negative_prompt_embeds_list.append(negative_prompt_embed)

            negative_prompt_embeds = torch.cat(negative_prompt_embeds_list, dim=0)

            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)
            # prompt_embeds = torch.cat([prompt_embeds, uncond_image_prompt_embeds], dim=1)
            # prompt_embeds = prompt_embeds
            # prompt_embeds = prompt_embeds.repeat(1, 1024, 1)
            # negative_prompt_embeds = image_prompt_embeds
            # negative_prompt_embeds = self.pipe.encode_prompt(
            #     negative_prompt,
            #     device=self.device,
            #     num_images_per_prompt=num_samples,
            #     do_classifier_free_guidance=True
            # )[0]

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images


class VgenXL(Vgen):
    """SDXL"""

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        self.generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            **kwargs,
        ).images

        return images


class VgenPlus(Vgen):
    """Vgen with fine-grained features"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=self.pipe.unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds


class VgenFull(VgenPlus):
    """Vgen with full features"""

    def init_proj(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model


class VgenPlusXL(Vgen):
    """SDXL"""

    def init_proj(self):
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.pipe.unet.config.cross_attention_dim,
            ff_mult=4,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float16)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        return image_prompt_embeds, uncond_image_prompt_embeds

    def generate(
            self,
            pil_image,
            prompt=None,
            negative_prompt=None,
            scale=1.0,
            num_samples=4,
            seed=None,
            num_inference_steps=30,
            **kwargs,
    ):
        self.set_scale(scale)

        num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)

        if prompt is None:
            prompt = "best quality, high quality"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(pil_image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

        with torch.inference_mode():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipe.encode_prompt(
                prompt,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds, uncond_image_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        images = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images

        return images

