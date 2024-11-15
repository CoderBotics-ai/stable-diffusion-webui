import torch
import math
import tqdm


class NoiseScheduleVP:
    def __init__(
        self,
        schedule: str = 'discrete',
        betas: torch.Tensor = None,
        alphas_cumprod: torch.Tensor = None,
        continuous_beta_0: float = 0.1,
        continuous_beta_1: float = 20.0,
    ):
        """Create a wrapper class for the forward SDE (VP type).

        This class supports both discrete-time and continuous-time diffusion models.

        Args:
            schedule (str): The noise schedule of the forward SDE. Options are 'discrete', 'linear', or 'cosine'.
            betas (torch.Tensor): The beta array for the discrete-time DPM.
            alphas_cumprod (torch.Tensor): The cumulative product of alphas for the discrete-time DPM.
            continuous_beta_0 (float): The smallest beta for the linear schedule.
            continuous_beta_1 (float): The largest beta for the linear schedule.
        """
        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError(f"Unsupported noise schedule {schedule}. Must be 'discrete', 'linear', or 'cosine'.")

        self.schedule = schedule
        if schedule == 'discrete':
            self._initialize_discrete_schedule(betas, alphas_cumprod)
        else:
            self._initialize_continuous_schedule(continuous_beta_0, continuous_beta_1)

    def _initialize_discrete_schedule(self, betas: torch.Tensor, alphas_cumprod: torch.Tensor):
        """Initialize parameters for discrete-time diffusion models."""
        if betas is not None:
            log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
        else:
            assert alphas_cumprod is not None, "Either betas or alphas_cumprod must be provided."
            log_alphas = 0.5 * torch.log(alphas_cumprod)

        self.total_N = len(log_alphas)
        self.T = 1.0
        self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1))
        self.log_alpha_array = log_alphas.reshape((1, -1))

    def _initialize_continuous_schedule(self, continuous_beta_0: float, continuous_beta_1: float):
        """Initialize parameters for continuous-time diffusion models."""
        self.total_N = 1000
        self.beta_0 = continuous_beta_0
        self.beta_1 = continuous_beta_1
        self.cosine_s = 0.008
        self.cosine_beta_max = 999.0
        self.cosine_t_max = math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi) * 2.0 * (1.0 + self.cosine_s) / math.pi - self.cosine_s
        self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0))
        self.schedule = 'cosine' if self.schedule == 'cosine' else 'linear'
        self.T = 0.9946 if self.schedule == 'cosine' else 1.0

    def marginal_log_mean_coeff(self, t: torch.Tensor) -> torch.Tensor:
        """Compute log(alpha_t) for a given continuous-time label t in [0, T]."""
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1,))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha_t for a given continuous-time label t in [0, T]."""
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sigma_t for a given continuous-time label t in [0, T]."""
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t: torch.Tensor) -> torch.Tensor:
        """Compute lambda_t = log(alpha_t) - log(sigma_t) for a given continuous-time label t in [0, T]."""
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb: torch.Tensor) -> torch.Tensor:
        """Compute the continuous-time label t for a given half-logSNR lambda_t."""
        if self.schedule == 'linear':
            tmp = 2.0 * (self.beta_1 - self.beta_0) * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2.0 * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]), torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2.0 * (1.0 + self.cosine_s) / math.pi - self.cosine_s
            return t_fn(log_alpha)


def model_wrapper(
    model,
    noise_schedule,
    model_type: str = "noise",
    model_kwargs: dict = None,
    guidance_type: str = "uncond",
    guidance_scale: float = 1.0,
    classifier_fn=None,
    classifier_kwargs: dict = None,
):
    """Create a wrapper function for the noise prediction model.

    This function wraps the model to accept continuous time as input and outputs the predicted noise.

    Args:
        model: A diffusion model.
        noise_schedule: A noise schedule object.
        model_type (str): The type of the diffusion model.
        model_kwargs (dict): Additional arguments for the model.
        guidance_type (str): The type of guidance for sampling.
        guidance_scale (float): The scale for guided sampling.
        classifier_fn: A classifier function for guidance.
        classifier_kwargs (dict): Additional arguments for the classifier function.

    Returns:
        A noise prediction model function.
    """
    model_kwargs = model_kwargs or {}
    classifier_kwargs = classifier_kwargs or {}

    def get_model_input_time(t_continuous: torch.Tensor) -> torch.Tensor:
        """Convert continuous-time `t_continuous` to model input time."""
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0
        return t_continuous

    def noise_pred_fn(x: torch.Tensor, t_continuous: torch.Tensor, cond=None) -> torch.Tensor:
        """Predict noise based on input and continuous time."""
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0],))
        t_input = get_model_input_time(t_continuous)
        output = model(x, t_input, cond, **model_kwargs) if cond else model(x, t_input, None, **model_kwargs)
        
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return (x - expand_dims(alpha_t, dims) * output) / expand_dims(sigma_t, dims)
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return expand_dims(alpha_t, dims) * output + expand_dims(sigma_t, dims) * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = x.dim()
            return -expand_dims(sigma_t, dims) * output

    def cond_grad_fn(x: torch.Tensor, t_input: torch.Tensor, condition) -> torch.Tensor:
        """Compute the gradient of the classifier."""
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x: torch.Tensor, t_continuous: torch.Tensor, condition, unconditional_condition) -> torch.Tensor:
        """The noise prediction model function used for DPM-Solver."""
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = t_continuous.expand((x.shape[0],))
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input, condition)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, dims=cond_grad.dim()) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1.0 or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                if isinstance(condition, dict):
                    assert isinstance(unconditional_condition, dict)
                    c_in = {k: torch.cat([unconditional_condition[k], condition[k]]) for k in condition}
                elif isinstance(condition, list):
                    c_in = [torch.cat([unconditional_condition[i], condition[i]]) for i in range(len(condition))]
                else:
                    c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class UniPC:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        predict_x0: bool = True,
        thresholding: bool = False,
        max_val: float = 1.0,
        variant: str = 'bh1',
        condition=None,
        unconditional_condition=None,
        before_sample=None,
        after_sample=None,
        after_update=None
    ):
        """Construct a UniPC.

        This class supports both data prediction and noise prediction.
        """
        self.model_fn_ = model_fn
        self.noise_schedule = noise_schedule
        self.variant = variant
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.max_val = max_val
        self.condition = condition
        self.unconditional_condition = unconditional_condition
        self.before_sample = before_sample
        self.after_sample = after_sample
        self.after_update = after_update

    def dynamic_thresholding_fn(self, x0: torch.Tensor, t=None) -> torch.Tensor:
        """Dynamic thresholding method."""
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def model(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Model prediction."""
        cond = self.condition
        uncond = self.unconditional_condition
        if self.before_sample is not None:
            x, t, cond, uncond = self.before_sample(x, t, cond, uncond)
        res = self.model_fn_(x, t, cond, uncond)
        if self.after_sample is not None:
            x, t, cond, uncond, res = self.after_sample(x, t, cond, uncond, res)

        if isinstance(res, tuple):
            res = res[1]  # Extract the predicted x0

        return res

    def noise_prediction_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the noise prediction model."""
        return self.model(x, t)

    def data_prediction_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the data prediction model (with thresholding)."""
        noise = self.noise_prediction_fn(x, t)
        dims = x.dim()
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - expand_dims(sigma_t, dims) * noise) / expand_dims(alpha_t, dims)
        if self.thresholding:
            p = 0.995  # Hyperparameter from the "Imagen" paper.
            s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
            s = expand_dims(torch.maximum(s, self.max_val * torch.ones_like(s).to(s.device)), dims)
            x0 = torch.clamp(x0, -s, s) / s
        return x0

    def model_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert the model to the noise prediction model or the data prediction model."""
        return self.data_prediction_fn(x, t) if self.predict_x0 else self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type: str, t_T: float, t_0: float, N: int, device) -> torch.Tensor:
        """Compute the intermediate time steps for sampling."""
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T ** (1. / t_order), t_0 ** (1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError(f"Unsupported skip_type {skip_type}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'")

    def get_orders_and_timesteps_for_singlestep_solver(self, steps: int, order: int, skip_type: str, t_T: float, t_0: float, device) -> tuple:
        """Get the order of each step for sampling by the singlestep DPM-Solver."""
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3] * (K - 1) + [1]
            else:
                orders = [3] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2] * K
            else:
                K = steps // 2 + 1
                orders = [2] * (K - 1) + [1]
        elif order == 1:
            K = steps
            orders = [1] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")

        if skip_type == 'logSNR':
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Denoise at the final step, equivalent to solving the ODE from lambda_s to infinity."""
        return self.data_prediction_fn(x, s)

    def multistep_uni_pc_update(self, x: torch.Tensor, model_prev_list: list, t_prev_list: list, t: torch.Tensor, order: int, **kwargs) -> tuple:
        """Perform a multistep update for the unified predictor-corrector method."""
        if len(t.shape) == 0:
            t = t.view(-1)
        if 'bh' in self.variant:
            return self.multistep_uni_pc_bh_update(x, model_prev_list, t_prev_list, t, order, **kwargs)
        else:
            assert self.variant == 'vary_coeff'
            return self.multistep_uni_pc_vary_update(x, model_prev_list, t_prev_list, t, order, **kwargs)

    def multistep_uni_pc_vary_update(self, x: torch.Tensor, model_prev_list: list, t_prev_list: list, t: torch.Tensor, order: int, use_corrector: bool = True) -> tuple:
        """Perform a varying coefficient update for the unified predictor-corrector method."""
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        # Compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=x.device)

        K = len(rks)
        # Build C matrix
        C = []

        col = torch.ones_like(rks)
        for k in range(1, K + 1):
            C.append(col)
            col = col * rks / (k + 1)
        C = torch.stack(C, dim=1)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            C_inv_p = torch.linalg.inv(C[:-1, :-1])
            A_p = C_inv_p

        if use_corrector:
            C_inv = torch.linalg.inv(C)
            A_c = C_inv

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_ks = []
        factorial_k = 1
        h_phi_k = h_phi_1
        for k in range(1, K + 2):
            h_phi_ks.append(h_phi_k)
            h_phi_k = h_phi_k / hh - 1 / factorial_k
            factorial_k *= (k + 1)

        model_t = None
        if self.predict_x0:
            x_t_ = (
                sigma_t / sigma_prev_0 * x
                - alpha_t * h_phi_1 * model_prev_0
            )
            # Predictor
            x_t = x_t_
            if len(D1s) > 0:
                # Compute residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * torch.einsum('bkchw,k->bchw', D1s, A_p[k])
            # Corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = (model_t - model_prev_0)
                x_t = x_t_
                for k in range(K - 1):
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * torch.einsum('bkchw,k->bchw', D1s, A_c[k][:-1])
                x_t = x_t - alpha_t * h_phi_ks[K] * (D1_t * A_c[k][-1])
        else:
            log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
            x_t_ = (
                (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                - (sigma_t * h_phi_1) * model_prev_0
            )
            # Predictor
            x_t = x_t_
            if len(D1s) > 0:
                # Compute residuals for predictor
                for k in range(K - 1):
                    x_t = x_t - sigma_t * h_phi_ks[k + 1] * torch.einsum('bkchw,k->bchw', D1s, A_p[k])
            # Corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = (model_t - model_prev_0)
                x_t = x_t_
                for k in range(K - 1):
                    x_t = x_t - sigma_t * h_phi_ks[k + 1] * torch.einsum('bkchw,k->bchw', D1s, A_c[k][:-1])
                x_t = x_t - sigma_t * h_phi_ks[K] * (D1_t * A_c[k][-1])
        return x_t, model_t