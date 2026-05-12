# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""
Map-style text-only SFT dataset for Tuna.

Reads JSONL files from a directory tree of class subdirectories.  Each line is
a JSON object with a ``conversations`` list of ``{"from": "human"/"gpt",
"value": "..."}`` dicts.  ``gpt`` (assistant) turns carry ``has_loss: 1``;
``human`` turns are conditioning only.

Returns the standard Tuna sample dict with ``data_type="mmu_text"``, dummy
image tensors, and the ``modality_positions`` sentinel ``[[-1, -1]]`` so the
model's ``_prepare_input`` skips image injection entirely.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any

import torch
from torch.utils.data import Dataset

from tuna.data.tokenize_utils import (
    format_conversation_und_text,
)
from tuna.data.transforms import build_image_transform


logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Map-style text-only SFT dataset over JSONL files in class subdirectories.

    Args:
        text_root: Root directory containing ``cls*`` subdirectories, each
            holding ``text_data_part_{i}.jsonl`` files.
        tokenizer: HuggingFace ``AutoTokenizer`` with Tuna special tokens.
        max_text_length: Maximum length of the unified token sequence.
        image_size: H/W used for the dummy ``images`` tensor.
        clip_image_size: H/W used for the dummy ``images_clip`` tensor.
        data_filename_format: Format string for JSONL filenames.  Must contain
            ``{}`` for the part index.
        cls_dirs: Explicit list of subdirectory names to scan.  When ``None``
            (default), all entries matching ``cls*`` under *text_root* are used.
        tuna_token_ids: Dict of special token ids from the model wrapper.
    """

    def __init__(
        self,
        text_root: str,
        tokenizer,
        max_text_length: int = 4096,
        image_size: int | tuple[int, int] = 512,
        clip_image_size: int | tuple[int, int] = 384,
        data_filename_format: str = "text_data_part_{}.jsonl",
        cls_dirs: list[str] | None = None,
        tuna_token_ids: dict[str, int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.text_root = text_root
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        )
        self.clip_image_size = (
            (clip_image_size, clip_image_size)
            if isinstance(clip_image_size, int) else tuple(clip_image_size)
        )
        self.data_filename_format = data_filename_format

        if cls_dirs is None:
            cls_dirs = sorted(
                d for d in os.listdir(text_root)
                if d.startswith("cls") and os.path.isdir(os.path.join(text_root, d))
            )
        self.cls_dirs = cls_dirs

        self.records: list[dict[str, Any]] = self._load_all_records()

        # Resolve special token ids.
        if tuna_token_ids is not None:
            self.bos_id = tuna_token_ids["bos_id"]
            self.eos_id = tuna_token_ids["eos_id"]
            self.pad_id = tokenizer.pad_token_id or self.eos_id
            self.boi_id = tuna_token_ids["boi_id"]
            self.eoi_id = tuna_token_ids["eoi_id"]
            self.img_pad_id = tuna_token_ids["img_pad_id"]
            self.img_id = tuna_token_ids.get("img_id", tuna_token_ids["img_pad_id"])
        else:
            raise ValueError("tuna_token_ids is required for TextDataset")

        # Pre-build the image transform for dummy image sizing (used in
        # images_clip fallback when no SigLIP is configured, which is always
        # the case for text-only).
        self.clip_image_transform = build_image_transform(self.clip_image_size, center_crop=True)

        logger.info(
            f"TextDataset: loaded {len(self.records)} records from {len(cls_dirs)} "
            f"class dirs under {text_root} (max_text_length={max_text_length})"
        )

    def _load_all_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for cls_dir in self.cls_dirs:
            cls_path = os.path.join(self.text_root, cls_dir)
            part_idx = 0
            while True:
                fname = self.data_filename_format.format(part_idx)
                fpath = os.path.join(cls_path, fname)
                if not os.path.isfile(fpath):
                    break
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            rec["_cls"] = cls_dir
                            records.append(rec)
                        except json.JSONDecodeError:
                            logger.warning(f"Skip malformed JSON in {fpath}")
                part_idx += 1
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _convert_to_messages(self, record: dict[str, Any]) -> list[dict[str, str]]:
        """Convert ``{from, value}`` conversations to HF chat-template format."""
        messages: list[dict[str, str]] = []
        role_map = {"human": "user", "gpt": "assistant"}
        for turn in record.get("conversations", []):
            role = turn.get("from") or turn.get("role") or "user"
            value = turn.get("value") or turn.get("content") or ""
            messages.append({"role": role_map.get(role, role), "content": value})
        return messages

    def _extract_sentence(self, record: dict[str, Any]) -> str:
        """Return the last assistant turn value for logging."""
        convs = record.get("conversations", [])
        for turn in reversed(convs):
            role = turn.get("from") or turn.get("role")
            if role in {"gpt", "assistant"}:
                return str(turn.get("value") or turn.get("content") or "")
        return ""

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]

        # 1. Build chat-templated token list.
        messages = self._convert_to_messages(record)
        try:
            token_ids = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
            )
        except Exception:
            # Fallback for tokenizers without a chat template.
            sentence = self._extract_sentence(record)
            token_ids = self.tokenizer(
                sentence, add_special_tokens=False, truncation=True,
                max_length=self.max_text_length,
            )["input_ids"]
        text_token_list = list(token_ids)

        # 2. Format as text-only conversation sequence.
        tt, tl, mp, tm, im = format_conversation_und_text(
            text_tokens=text_token_list,
            eos_id=self.eos_id,
            boi_id=self.boi_id,
            eoi_id=self.eoi_id,
            pad_id=self.pad_id,
            img_id=self.img_id,
            img_pad_id=self.img_pad_id,
            num_image_tokens=0,
            max_seq_len=self.max_text_length,
        )

        sentence = self._extract_sentence(record)

        H, W = self.image_size
        clip_H, clip_W = self.clip_image_size

        return {
            "images": torch.zeros(3, H, W),
            "text_tokens": tt.long(),
            "text_labels": tl.long(),
            "text_masks": tm.bool(),
            "image_masks": torch.zeros(self.max_text_length, dtype=torch.bool),
            "modality_positions": mp.long(),
            "data_type": "mmu_text",
            "sentence": sentence,
            "images_clip": torch.zeros(3, clip_H, clip_W),
        }
