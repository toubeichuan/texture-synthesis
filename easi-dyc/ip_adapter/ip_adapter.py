import os

import torch
import torch.nn as nn
from diffusers.models.lora import LoRALinearLayer
from diffusers.pipelines.controlnet import MultiControlNetModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image

if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    from .attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, CNAttnProcessor2_0 as CNAttnProcessor
else:
    from .attention_processor import IPAttnProcessor, AttnProcessor, CNAttnProcessor
from .resampler import Resampler


class IPLoRAAttnProcessor(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism using PyTorch 2.0's memory-efficient scaled dot-product
    attention.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
        kwargs (`dict`):
            Additional keyword arguments to pass to the `LoRALinearLayer` layers.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim=None,
        rank: int = 4,
        network_alpha= None,
        ip_processor=None,
        **kwargs,
    ):
        super().__init__()
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("IPLoRAAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        q_rank = kwargs.pop("q_rank", None)
        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_rank = q_rank if q_rank is not None else rank
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size

        v_rank = kwargs.pop("v_rank", None)
        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_rank = v_rank if v_rank is not None else rank
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size

        out_rank = kwargs.pop("out_rank", None)
        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_rank = out_rank if out_rank is not None else rank
        out_hidden_size = out_hidden_size if out_hidden_size is not None else hidden_size

        self.to_q_lora = LoRALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(out_hidden_size, out_hidden_size, out_rank, network_alpha)

        self.ip_processor = ip_processor  # to be assigned at runtime
        self.to_k_ip_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_ip_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)

    def __call__(self, attn, hidden_states, *args, **kwargs) -> torch.FloatTensor:
        attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
        attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
        attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
        attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

        assert self.ip_processor is not None
        attn._modules.pop("processor")
        attn.processor = self.ip_processor.to(hidden_states.device)
        attn.processor.to_k_ip.lora_layer = self.to_k_ip_lora.to(hidden_states.device)
        attn.processor.to_v_ip.lora_layer = self.to_v_ip_lora.to(hidden_states.device)
        out = attn.processor(attn, hidden_states, *args, **kwargs)
        return out


class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class IPAdapter:
    def __init__(self, pipe, ipadapter_ckpt_path, image_encoder_path, device="cuda", dtype=torch.float16, resample=Image.Resampling.LANCZOS):
        self.pipe = pipe
        self.device = device
        self.dtype = dtype
        
        # load ip adapter model
        ipadapter_model = torch.load(ipadapter_ckpt_path, map_location="cpu")

        # detect features
        self.is_plus = "latents" in ipadapter_model["image_proj"]
        self.output_cross_attention_dim = ipadapter_model["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        self.is_sdxl = self.output_cross_attention_dim == 2048
        self.cross_attention_dim = 1280 if self.is_plus and self.is_sdxl else self.output_cross_attention_dim
        self.heads = 20 if self.is_sdxl and self.is_plus else 12
        self.num_tokens = 16 if self.is_plus else 4

        # set image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(self.device, dtype=self.dtype)
        self.clip_image_processor = CLIPImageProcessor(resample=resample)

        # set IPAdapter
        self.set_ip_adapter()
        self.image_proj_model = self.init_proj() if not self.is_plus else self.init_proj_plus()
        self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(ipadapter_model["ip_adapter"])
        
    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=self.dtype)
        return image_proj_model
    
    def init_proj_plus(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=self.heads,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4
        ).to(self.device, dtype=self.dtype)
        return image_proj_model

    def set_ip_adapter(self):
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
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim).to(self.device, dtype=self.dtype)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor())
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor())
    
    @torch.inference_mode()
    def get_image_embeds(self, images, negative_images=None):
        clip_image = self.clip_image_processor(images=images, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=self.dtype)

        if not self.is_plus:
            clip_image_embeds = self.image_encoder(clip_image).image_embeds
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            if negative_images is not None:
                negative_clip_image = self.clip_image_processor(images=negative_images, return_tensors="pt").pixel_values
                negative_clip_image = negative_clip_image.to(self.device, dtype=self.dtype)
                negative_image_prompt_embeds = self.image_encoder(negative_clip_image).image_embeds
            else:
                negative_image_prompt_embeds = torch.zeros_like(clip_image_embeds)
            negative_image_prompt_embeds = self.image_proj_model(negative_image_prompt_embeds)
        else:
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            if negative_images is not None:
                negative_clip_image = self.clip_image_processor(images=negative_images, return_tensors="pt").pixel_values
                negative_clip_image = negative_clip_image.to(self.device, dtype=self.dtype)
                negative_clip_image_embeds = self.image_encoder(negative_clip_image, output_hidden_states=True).hidden_states[-2]
            else:
                negative_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
            negative_image_prompt_embeds = self.image_proj_model(negative_clip_image_embeds)
        
        num_tokens = image_prompt_embeds.shape[0] * self.num_tokens
        self.set_tokens(num_tokens)

        return image_prompt_embeds, negative_image_prompt_embeds

    @torch.inference_mode()
    def get_prompt_embeds(self, images, negative_images=None, prompt=None, negative_prompt=None, weight=[]):
        prompt_embeds, negative_prompt_embeds = self.get_image_embeds(images, negative_images=negative_images)

        if any(e != 1.0 for e in weight):
            weight = torch.tensor(weight).unsqueeze(-1).unsqueeze(-1)
            weight = weight.to(self.device)
            prompt_embeds = prompt_embeds * weight

        if prompt_embeds.shape[0] > 1:
            prompt_embeds = torch.cat(prompt_embeds.chunk(prompt_embeds.shape[0]), dim=1)
        if negative_prompt_embeds.shape[0] > 1:
            negative_prompt_embeds = torch.cat(negative_prompt_embeds.chunk(negative_prompt_embeds.shape[0]), dim=1)

        text_embeds = (None, None, None, None)
        if prompt is not None:
            text_embeds = self.pipe.encode_prompt(
                prompt,
                negative_prompt=negative_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            prompt_embeds = torch.cat((text_embeds[0], prompt_embeds), dim=1)
            negative_prompt_embeds = torch.cat((text_embeds[1], negative_prompt_embeds), dim=1)

        output = (prompt_embeds, negative_prompt_embeds)

        if self.is_sdxl:
            output += (text_embeds[2], text_embeds[3])
        
        return output
    
    @staticmethod
    def combine_text_image_embeds(image_embeds, negative_image_embeds, text_embeds, negative_text_embeds, weight=[]):
        """
        Custom function to combine text and image embedings. Currently does not support SD-XL.
        """
        # Assert none of the inputs is None
        for ele in (image_embeds, negative_image_embeds, text_embeds, negative_text_embeds):
            assert ele is not None
        
        if any(e != 1.0 for e in weight):
            weight = torch.tensor(weight).unsqueeze(-1).unsqueeze(-1)
            weight = weight.to(image_embeds.device)
            image_embeds = image_embeds * weight

        if image_embeds.shape[0] > 1:
            image_embeds = torch.cat(image_embeds.chunk(image_embeds.shape[0]), dim=1)
        if negative_image_embeds.shape[0] > 1:
            negative_image_embeds = torch.cat(negative_image_embeds.chunk(negative_image_embeds.shape[0]), dim=1)
        
        # Combine text and image embeddings
        prompt_embeds = torch.cat((text_embeds, image_embeds), dim=1)
        negative_prompt_embeds = torch.cat((negative_text_embeds, negative_image_embeds), dim=1)

        output = (prompt_embeds, negative_prompt_embeds)
        
        return output

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale
            elif isinstance(attn_processor, IPLoRAAttnProcessor):
                attn_processor.ip_processor.scale = scale
    
    def set_tokens(self, num_tokens):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.num_tokens = num_tokens
            elif isinstance(attn_processor, IPLoRAAttnProcessor):
                attn_processor.ip_processor.num_tokens = num_tokens
        
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    for attn_processor in controlnet.attn_processors.values():
                        if isinstance(attn_processor, CNAttnProcessor):
                            attn_processor.num_tokens = num_tokens
            else:
                for attn_processor in self.pipe.controlnet.attn_processors.values():
                    if isinstance(attn_processor, CNAttnProcessor):
                        attn_processor.num_tokens = num_tokens


def _load_ipadapter_state(ip_adapter_path: str):
    ckpt_candidates = [
        os.path.join(ip_adapter_path, "ip-adapter-plus_sd15.bin"),
        os.path.join(ip_adapter_path, "ip-adapter-plus_sd15.safetensors"),
    ]
    for ckpt_path in ckpt_candidates:
        if os.path.exists(ckpt_path):
            if ckpt_path.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise ImportError(
                        "safetensors 未安装，无法加载 .safetensors 格式的 IP-Adapter 权重"
                    ) from exc
                return load_file(ckpt_path)
            return torch.load(ckpt_path, map_location="cpu")
    raise FileNotFoundError(
        f"未找到 IP-Adapter 权重文件，期望位置之一：{ckpt_candidates}"
    )


def setup_ipadapter_attention_processors(
    pipe,
    num_tokens,
    device,
    ip_adapter_weights=None,
    ip_adapter_path=None,
    dtype=torch.float16,
):
    """
    为 Stable Diffusion pipeline 设置 IP-Adapter 的 attention processor。

    Args:
        pipe: 已初始化的 StableDiffusion pipeline（含 UNet 与可选 ControlNet）
        num_tokens: IP-Adapter 输出 token 数
        device: torch.device
        ip_adapter_weights: 预加载的 IP-Adapter 权重 (ip_adapter state_dict)
        ip_adapter_path: 当未提供 ip_adapter_weights 时，权重所在路径
        dtype: attention processor 使用的数据类型
    """
    
    if ip_adapter_weights is None:
        if ip_adapter_path is None:
            raise ValueError("必须提供 ip_adapter_weights 或 ip_adapter_path 之一")
        ip_adapter_state = _load_ipadapter_state(ip_adapter_path)
        ip_adapter_weights = ip_adapter_state.get("ip_adapter", {})
    elif isinstance(ip_adapter_weights, str):
        # 兼容传入单独权重文件路径的情况
        ip_adapter_state = torch.load(ip_adapter_weights, map_location="cpu")
        ip_adapter_weights = ip_adapter_state.get("ip_adapter", {})

    unet = pipe.unet
    attn_procs = {}
    ip_layer_names = []
    # 预先缓存 block 通道配置
    block_out_channels = list(unet.config.block_out_channels)
    up_block_channels = list(reversed(block_out_channels))

    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        )

        if name.startswith("mid_block"):
            hidden_size = block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = up_block_channels[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = block_out_channels[block_id]
        else:
            hidden_size = block_out_channels[0]

        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            proc = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                num_tokens=num_tokens,
            ).to(device, dtype=dtype)
            attn_procs[name] = proc
            ip_layer_names.append(name)

    unet.set_attn_processor(attn_procs)

    if ip_adapter_weights:
        # 直接使用 diffusers 注册的顺序来加载，避免层顺序不匹配
        module_list = torch.nn.ModuleList(unet.attn_processors.values())
        incompatible = module_list.load_state_dict(ip_adapter_weights, strict=False)
        # load_state_dict returns an IncompatibleKeys object with missing_keys / unexpected_keys
        missing_keys = list(getattr(incompatible, "missing_keys", []))
        unexpected_keys = list(getattr(incompatible, "unexpected_keys", []))

        def _trim(keys, limit=50):
            if len(keys) <= limit:
                return keys
            return keys[:limit] + [f"... ({len(keys) - limit} more)"]

        print("[IP-Adapter] load_state_dict strict=False")
        print(f"[IP-Adapter] missing_keys ({len(missing_keys)}): {_trim(missing_keys)}")
        print(f"[IP-Adapter] unexpected_keys ({len(unexpected_keys)}): {_trim(unexpected_keys)}")

    # Sanity: count how many IP attention processors are installed on the UNet
    try:
        ip_cnt = 0
        for p in pipe.unet.attn_processors.values():
            if isinstance(p, IPAttnProcessor):
                ip_cnt += 1
        print(f"[IP-Adapter] UNet IPAttnProcessor count: {ip_cnt} / total {len(pipe.unet.attn_processors)}")
    except Exception as e:
        print(f"[IP-Adapter] UNet IPAttnProcessor count: <failed> ({type(e).__name__}: {e})")

    if hasattr(pipe, "controlnet"):
        if isinstance(pipe.controlnet, MultiControlNetModel):
            for controlnet in pipe.controlnet.nets:
                cn_attn_procs = {}
                for name in controlnet.attn_processors.keys():
                    cn_attn_procs[name] = CNAttnProcessor(num_tokens=num_tokens)
                controlnet.set_attn_processor(cn_attn_procs)
        else:
            cn_attn_procs = {}
            for name in pipe.controlnet.attn_processors.keys():
                cn_attn_procs[name] = CNAttnProcessor(num_tokens=num_tokens)
            pipe.controlnet.set_attn_processor(cn_attn_procs)



