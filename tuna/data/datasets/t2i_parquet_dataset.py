# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""
Map-style T2I dataset backed by Parquet metadata + Zip image archives.

Configured via a YAML file whose ``info`` key is a list of
``[parquet_root, zip_root, prompt_keys]`` entries.  Each parquet file
contains rows with an ``ID`` column and one column per prompt key.  The
corresponding zip archive stores the image bytes under the entry name
``{ID}`` (no file extension).

At training time a prompt key is randomly picked per sample and used as
the text conditioning for the T2I generation task.
"""

from __future__ import annotations

import glob
import io
import logging
import os
import random
import zipfile
from functools import lru_cache
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from tuna.data.tokenize_utils import format_sequence_gen_qwen2_5
from tuna.data.transforms import build_image_transform, build_siglip_transform


logger = logging.getLogger(__name__)


class T2IParquetDataset(Dataset):
    """Map-style T2I dataset over Parquet + Zip archives.

    Args:
        config_yaml: Path to a YAML file with an ``info`` list of
            ``[parquet_root, zip_root, prompt_keys]`` triples.  Paths are
            resolved relative to the YAML file's directory.
        tokenizer: HuggingFace ``AutoTokenizer`` with Tuna special tokens.
        image_size: Target ``(H, W)`` for the WAN-VAE / patch input.
        max_text_length: Maximum length of the unified token sequence.
        center_crop: Passed through to ``build_image_transform``.
        num_image_tokens: How many ``<img_pad>`` tokens per image.  Defaults
            to ``(H/16) * (W/16)``.
        siglip_processor_id: If set, emit SigLIP2 inputs for the image.
        clip_image_size: Fallback ``images_clip`` resolution when SigLIP is
            not configured.
        tuna_token_ids: Dict of special token ids from the model wrapper.
    """

    def __init__(
        self,
        config_yaml: str,
        tokenizer,
        image_size: int | tuple[int, int] = 256,
        max_text_length: int = 256,
        center_crop: bool = True,
        num_image_tokens: int | None = None,
        siglip_processor_id: str | None = None,
        clip_image_size: int | tuple[int, int] = 384,
        tuna_token_ids: dict[str, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config_yaml = config_yaml
        self.tokenizer = tokenizer
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        )
        self.max_text_length = max_text_length

        # Resolve special token ids.
        if tuna_token_ids is not None:
            self.bos_id = tuna_token_ids["bos_id"]
            self.eos_id = tuna_token_ids["eos_id"]
            self.pad_id = tokenizer.pad_token_id or self.eos_id
            self.boi_id = tuna_token_ids["boi_id"]
            self.eoi_id = tuna_token_ids["eoi_id"]
            self.img_pad_id = tuna_token_ids["img_pad_id"]
        else:
            raise ValueError("tuna_token_ids is required for T2IParquetDataset")

        if num_image_tokens is None:
            patch = 16
            num_image_tokens = (self.image_size[0] // patch) * (self.image_size[1] // patch)
        self.num_image_tokens = num_image_tokens

        # Image transforms.
        self.image_transform = build_image_transform(self.image_size, center_crop)
        self.siglip_processor_id = siglip_processor_id
        self.siglip_transform = (
            build_siglip_transform(siglip_processor_id)
            if siglip_processor_id is not None
            else None
        )
        self.clip_image_size = (
            (clip_image_size, clip_image_size)
            if isinstance(clip_image_size, int)
            else tuple(clip_image_size)
        )
        self.clip_image_transform = build_image_transform(self.clip_image_size, center_crop)

        # Parse YAML config and build flat sample index.
        self.samples: list[dict[str, Any]] = self._build_index()

        logger.info(
            f"T2IParquetDataset: loaded {len(self.samples)} samples from "
            f"{config_yaml} (image_size={self.image_size}, "
            f"num_image_tokens={self.num_image_tokens})"
        )

    # ---- config parsing ----------------------------------------------------

    def _parse_yaml_config(self) -> list[tuple[str, str, list[str]]]:
        with open(self.config_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        yaml_dir = os.path.dirname(os.path.abspath(self.config_yaml))
        entries: list[tuple[str, str, list[str]]] = []
        for item in cfg.get("info", []):
            parquet_rel, zip_rel, prompt_keys = item[0], item[1], item[2]
            parquet_root = os.path.join(yaml_dir, parquet_rel)
            zip_root = os.path.join(yaml_dir, zip_rel)
            entries.append((parquet_root, zip_root, list(prompt_keys)))
        return entries

    # ---- sample index ------------------------------------------------------

    def _build_index(self) -> list[dict[str, Any]]:
        import pandas as pd

        entries = self._parse_yaml_config()
        samples: list[dict[str, Any]] = []
        for parquet_root, zip_root, prompt_keys in entries:
            parquet_files = sorted(glob.glob(os.path.join(parquet_root, "data_batch_*.parquet")))
            for pq_path in parquet_files:
                # Match corresponding zip file.
                basename = os.path.splitext(os.path.basename(pq_path))[0]
                zip_path = os.path.join(zip_root, f"{basename}.zip")
                if not os.path.isfile(zip_path):
                    logger.warning(f"Zip file not found for {pq_path}, skipping")
                    continue

                df = pd.read_parquet(pq_path)
                for _, row in df.iterrows():
                    sample = {
                        "id": str(row["ID"]),
                        "zip_path": zip_path,
                        "prompt_keys": prompt_keys,
                        "row": row.to_dict(),
                    }
                    samples.append(sample)
        return samples

    # ---- image loading -----------------------------------------------------

    @staticmethod
    @lru_cache(maxsize=32)
    def _open_zip(zip_path: str) -> zipfile.ZipFile:
        return zipfile.ZipFile(zip_path, "r")

    def _load_image(self, sample: dict[str, Any]) -> Image.Image:
        zip_path = sample["zip_path"]
        sample_id = sample["id"]
        zf = self._open_zip(zip_path)
        data = zf.read(sample_id)
        img = Image.open(io.BytesIO(data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    # ---- text extraction ---------------------------------------------------

    def _extract_prompt(self, sample: dict[str, Any]) -> str:
        prompt_keys = sample["prompt_keys"]
        key = random.choice(prompt_keys)
        row = sample["row"]
        return str(row.get(key, row.get("caption1", "")))

    def _tokenize_prompt(self, prompt: str) -> list[int]:
        reserve = 1 + 2 + self.num_image_tokens + 1  # bos + boi/eoi + img_pads + eos
        max_text = max(0, self.max_text_length - reserve)
        ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max(1, max_text),
        )["input_ids"]
        return list(ids)

    # ---- main --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        # 1. Load and transform image.
        pil = self._load_image(sample)
        image_tensor = self.image_transform(pil)

        # 2. Extract and tokenize prompt.
        prompt = self._extract_prompt(sample)
        text_token_ids = self._tokenize_prompt(prompt)

        # 3. Format T2I sequence (all text_labels = -100, image loss via flow head).
        tt, tl, mp, tm, im = format_sequence_gen_qwen2_5(
            text_tokens=text_token_ids,
            system_tokens=None,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            boi_id=self.boi_id,
            eoi_id=self.eoi_id,
            pad_id=self.pad_id,
            img_pad_id=self.img_pad_id,
            num_image_tokens=self.num_image_tokens,
            max_seq_len=self.max_text_length,
            system_token_len=0,
        )

        out: dict[str, Any] = {
            "images": image_tensor,
            "text_tokens": tt.long(),
            "text_labels": tl.long(),
            "text_masks": tm.bool(),
            "image_masks": im.bool(),
            "modality_positions": mp.long(),
            "data_type": "t2i",
            "sentence": prompt,
        }

        # 4. Build images_clip.
        if self.siglip_transform is not None:
            sig = self.siglip_transform(pil)
            out["images_clip"] = sig["pixel_values"]
            out["siglip_pixel_attention_mask"] = sig["pixel_attention_mask"]
            out["siglip_spatial_shapes"] = sig["spatial_shapes"]
        else:
            out["images_clip"] = self.clip_image_transform(pil)

        return out
