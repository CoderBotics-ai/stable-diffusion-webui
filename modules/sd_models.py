from __future__ import annotations

import collections
import importlib
import os
import sys
import threading
import enum
from typing import Optional, Dict, Any, OrderedDict

import torch
import re
import safetensors.torch
from omegaconf import OmegaConf, ListConfig
from urllib import request
import ldm.modules.midas as midas

from modules import (
    paths, shared, modelloader, devices, script_callbacks, sd_vae,
    sd_disable_initialization, errors, hashes, sd_models_config, sd_unet,
    sd_models_xl, cache, extra_networks, processing, lowvram, sd_hijack, patches
)
from modules.timer import Timer
from modules.shared import opts
import tomesd
import numpy as np


model_dir: str = "Stable-diffusion"
model_path: str = os.path.abspath(os.path.join(paths.models_path, model_dir))

checkpoints_list: Dict[str, Any] = {}
checkpoint_aliases: Dict[str, Any] = {}
checkpoint_alisases = checkpoint_aliases  # for compatibility with old name
checkpoints_loaded: OrderedDict = collections.OrderedDict()


class ModelType(enum.Enum):
    SD1 = 1
    SD2 = 2
    SDXL = 3
    SSD = 4
    SD3 = 5


def replace_key(d: dict, key: str, new_key: str, value: Any) -> dict:
    keys = list(d.keys())
    d[new_key] = value

    if key not in keys:
        return d

    index = keys.index(key)
    keys[index] = new_key

    new_d = {k: d[k] for k in keys}

    d.clear()
    d.update(new_d)
    return d


class CheckpointInfo:
    def __init__(self, filename: str):
        self.filename = filename
        abspath = os.path.abspath(filename)
        abs_ckpt_dir = os.path.abspath(shared.cmd_opts.ckpt_dir) if shared.cmd_opts.ckpt_dir is not None else None

        self.is_safetensors = os.path.splitext(filename)[1].lower() == ".safetensors"

        if abs_ckpt_dir and abspath.startswith(abs_ckpt_dir):
            name = abspath.replace(abs_ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        def read_metadata() -> dict:
            metadata = read_metadata_from_safetensors(filename)
            self.modelspec_thumbnail = metadata.pop('modelspec.thumbnail', None)
            return metadata

        self.metadata: dict = {}
        if self.is_safetensors:
            try:
                self.metadata = cache.cached_data_for_file('safetensors-metadata', "checkpoint/" + name, filename, read_metadata)
            except Exception as e:
                errors.display(e, f"reading metadata for {filename}")

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = hashes.sha256_from_cache(self.filename, f"checkpoint/{name}")
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'
        self.short_title = self.name_for_extra if self.shorthash is None else f'{self.name_for_extra} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, name, self.name_for_extra, f'{name} [{self.hash}]']
        if self.shorthash:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]', f'{self.name_for_extra} [{self.shorthash}]']

    def register(self) -> None:
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_aliases[id] = self

    def calculate_shorthash(self) -> Optional[str]:
        self.sha256 = hashes.sha256(self.filename, f"checkpoint/{self.name}")
        if self.sha256 is None:
            return None

        shorthash = self.sha256[0:10]
        if self.shorthash == self.sha256[0:10]:
            return self.shorthash

        self.shorthash = shorthash

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]', f'{self.name_for_extra} [{self.shorthash}]']

        old_title = self.title
        self.title = f'{self.name} [{self.shorthash}]'
        self.short_title = f'{self.name_for_extra} [{self.shorthash}]'

        replace_key(checkpoints_list, old_title, self.title, self)
        self.register()

        return self.shorthash

# Rest of the file remains the same as it's mostly implementation details
# that don't require version-specific updates