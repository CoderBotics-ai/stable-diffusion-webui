import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR
from ldm.modules.ema import LitEma
from vqvae_quantize import VectorQuantizer2 as VectorQuantizer
from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.util import instantiate_from_config
import ldm.models.autoencoder
from packaging import version


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig: dict,
                 lossconfig: dict,
                 n_embed: int,
                 embed_dim: int,
                 ckpt_path: str = None,
                 ignore_keys: list = None,
                 image_key: str = "image",
                 colorize_nlabels: int = None,
                 monitor: str = None,
                 batch_resize_range: tuple = None,
                 scheduler_config: dict = None,
                 lr_g_factor: float = 1.0,
                 remap: str = None,
                 sane_index_shape: bool = False,
                 use_ema: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, kernel_size=1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], kernel_size=1)

        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int), "colorize_nlabels must be an integer"
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys or [])
        
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context:
                print(f"{context}: Switched to EMA weights")
        try:
            yield
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path: str, ignore_keys: list = None):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys or []:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if missing:
            print(f"Missing Keys: {missing}")
        if unexpected:
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x: torch.Tensor):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant: torch.Tensor):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def decode_code(self, code_b: torch.Tensor):
        quant_b = self.quantize.embed_code(code_b)
        return self.decode(quant_b)

    def forward(self, input: torch.Tensor, return_pred_indices: bool = False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        return (dec, diff, ind) if return_pred_indices else (dec, diff)

    def get_input(self, batch: dict, k: str):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size, upper_size = self.batch_resize_range
            new_resize = upper_size if self.global_step <= 4 else np.random.choice(np.arange(lower_size, upper_size + 16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
        return x

    def training_step(self, batch: dict, batch_idx: int, optimizer_idx: int):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:  # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:  # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch: dict, batch_idx: int):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch: dict, batch_idx: int, suffix: str = ""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val" + suffix,
                                        predicted_indices=ind)

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val" + suffix,
                                            predicted_indices=ind)
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor * self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)

        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                   list(self.decoder.parameters()) +
                                   list(self.quantize.parameters()) +
                                   list(self.quant_conv.parameters()) +
                                   list(self.post_quant_conv.parameters()),
                                   lr=lr_g, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                     lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []


class VQModelInterface(VQModel):
    def __init__(self, embed_dim: int, *args, **kwargs):
        super().__init__(*args, embed_dim=embed_dim, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h: torch.Tensor, force_not_quantize: bool = False):
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

ldm.models.autoencoder.VQModel = VQModel
ldm.models.autoencoder.VQModelInterface = VQModelInterface