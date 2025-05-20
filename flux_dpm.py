import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import FluxPipeline

from dpm_solver_pytorch import DPM_Solver


class NoiseScheduleFM:
    def __init__(
        self,
    ):
        self.T = 1. - 1e-6
        self.total_N = 1000

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return torch.log(self.marginal_alpha(t))

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return 1 - t.float()

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return t.float()

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        # return self.marginal_log_mean_coeff(t) - torch.log(self.marginal_std(t))
        return torch.log(self.marginal_alpha(t) / self.marginal_std(t))

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        return 1 / (1 + torch.exp(lamb.float()))


@torch.no_grad()
def dpm_fm_sample(
    self: FluxPipeline,
    dpm_args,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    true_cfg_scale: float = 1.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    # ip_adapter_image: Optional[PipelineImageInput] = None,
    # ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    # negative_ip_adapter_image: Optional[PipelineImageInput] = None,
    # negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
):
    assert not return_dict
    assert true_cfg_scale == 1.0
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
    )
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if do_true_cfg:
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            _,
        ) = self.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
            prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    if self.joint_attention_kwargs is None:
        self._joint_attention_kwargs = {}

    def model_fn(xt, t):
        t = t.to(torch.bfloat16)
        vf = self.transformer(
            hidden_states=xt,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]
        epsilon = xt + (1 - t) * vf
        return epsilon

    schedule = NoiseScheduleFM()
    solver = DPM_Solver(
        model_fn=model_fn,
        noise_schedule=schedule,
        algorithm_type=dpm_args.dpm_algorithm_type,  # "dpmsolver++",
    )

    latents = solver.sample(
        x=latents,
        steps=num_inference_steps,
        order=dpm_args.dpm_order,  # 1
        skip_type=dpm_args.dpm_skip_type,  # "time_uniform",
        method=dpm_args.dpm_method,  # "multistep",  # "singlestep",
    )

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    return image


@dataclass
class DPMArgs:
    dpm_order: int = 3
    dpm_skip_type: str = "logSNR"
    dpm_method: str = "singlestep"
    dpm_algorithm_type: str = "dpmsolver++"


def main():
    num_inference_steps = int(sys.argv[1])

    ckpt = 'black-forest-labs/FLUX.1-dev'
    pipe = FluxPipeline.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to("cuda")
    prompt = "A cat holding a sign that says hello world"
    generator = torch.Generator("cpu").manual_seed(0)
    dpm_args = DPMArgs(
        dpm_order=3,
        dpm_skip_type='logSNR',
        dpm_method='singlestep',
        dpm_algorithm_type='dpmsolver++',
    )
    image = dpm_fm_sample(
        pipe,
        dpm_args=dpm_args,
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=generator,
        return_dict=False,
    )[0]

    image.save(f'flux-dpm-{num_inference_steps}.png')


if __name__ == "__main__":
    main()
