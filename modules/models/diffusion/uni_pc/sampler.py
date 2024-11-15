import torch
from .uni_pc import NoiseScheduleVP, model_wrapper, UniPC
from modules import shared, devices


class UniPCSampler:
    """Sampler for the UniPC model, responsible for generating samples using the specified model and noise schedule."""

    def __init__(self, model, **kwargs):
        """Initialize the UniPCSampler with a model and optional parameters."""
        super().__init__()
        self.model = model
        self.before_sample = None
        self.after_sample = None
        self.register_buffer('alphas_cumprod', self._to_torch(model.alphas_cumprod))

    def _to_torch(self, tensor):
        """Convert a tensor to float32 and move it to the model's device."""
        return tensor.clone().detach().to(torch.float32).to(self.model.device)

    def register_buffer(self, name, attr):
        """Register a buffer for the sampler, ensuring it is on the correct device."""
        if isinstance(attr, torch.Tensor) and attr.device != devices.device:
            attr = attr.to(devices.device)
        setattr(self, name, attr)

    def set_hooks(self, before_sample, after_sample, after_update):
        """Set hooks for actions to be performed before and after sampling."""
        self.before_sample = before_sample
        self.after_sample = after_sample
        self.after_update = after_update

    @torch.no_grad()
    def sample(self, S, batch_size, shape, conditioning=None, **kwargs):
        """Generate samples from the model using the specified parameters."""
        self._validate_conditioning(conditioning, batch_size)

        # Prepare the image tensor
        img = self._initialize_image_tensor(batch_size, shape)

        # Initialize noise schedule
        ns = NoiseScheduleVP('discrete', alphas_cumprod=self.alphas_cumprod)

        # Determine model type based on parameterization
        model_type = "v" if self.model.parameterization == "v" else "noise"

        # Create model function wrapper
        model_fn = model_wrapper(
            lambda x, t, c: self.model.apply_model(x, t, c),
            ns,
            model_type=model_type,
            guidance_type="classifier-free",
            guidance_scale=kwargs.get('unconditional_guidance_scale', 1.0),
        )

        # Create UniPC instance and sample
        uni_pc = UniPC(
            model_fn,
            ns,
            predict_x0=True,
            thresholding=False,
            variant=shared.opts.uni_pc_variant,
            condition=conditioning,
            unconditional_condition=kwargs.get('unconditional_conditioning'),
            before_sample=self.before_sample,
            after_sample=self.after_sample,
            after_update=self.after_update
        )
        return uni_pc.sample(img, steps=S, skip_type=shared.opts.uni_pc_skip_type,
                              method="multistep", order=shared.opts.uni_pc_order,
                              lower_order_final=shared.opts.uni_pc_lower_order_final)

    def _validate_conditioning(self, conditioning, batch_size):
        """Validate the conditioning input to ensure it matches the expected batch size."""
        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list):
                    ctmp = ctmp[0]
                if ctmp.shape[0] != batch_size:
                    print(f"Warning: Got {ctmp.shape[0]} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {ctmp.shape[0]} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

    def _initialize_image_tensor(self, batch_size, shape):
        """Initialize the image tensor for sampling."""
        C, H, W = shape
        size = (batch_size, C, H, W)
        device = self.model.betas.device
        return torch.randn(size, device=device) if kwargs.get('x_T') is None else kwargs['x_T']