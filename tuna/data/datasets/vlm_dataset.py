# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""
Map-style VLM dataset backed by JSONL annotations + TAR image archives.

Each JSONL line describes one sample:

.. code-block:: json

    {"id": "...",
     "data": [
       {"role": "user", "content": [
         {"type": "image", "image": {"relative_path": "img_1_3.jpg"}},
         {"type": "text", "text": {"string": "What is in this image?"}}
       ]},
       {"role": "assistant", "content": [
         {"type": "text", "text": {"string": "A cat."}}
       ]}
     ]}

Images are stored in TAR archives under *image_root* and referenced by
``relative_path``.  The dataset builds an in-memory TAR member index so
random access via ``tarfile.extractfile`` is fast.

The assistant text is the LM loss target; user text and images are
conditioning only.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tarfile
from functools import lru_cache
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from tuna.data.tokenize_utils import format_sequence_und
from tuna.data.transforms import build_image_transform, build_siglip_transform


logger = logging.getLogger(__name__)


class VLMDataset(Dataset):
    """Map-style VLM dataset over JSONL + TAR archives.

    Args:
        annotation_dir: Directory containing ``data_part_{i}.jsonl`` files.
        image_root: Directory containing ``batch_{i}.tar`` files.
        tokenizer: HuggingFace ``AutoTokenizer`` with Tuna special tokens.
        image_size: Target ``(H, W)`` for the image tensor.
        max_text_length: Maximum length of the unified token sequence.
        data_filename_format: Format string for annotation filenames.
        range_start: First part index (inclusive).
        range_end: Last part index (exclusive).
        sample_num: Optional hard cap on number of samples to load.
        center_crop: Passed through to ``build_image_transform``.
        num_image_tokens: How many ``<img_pad>`` tokens per image.
            Defaults to ``(H/16) * (W/16)``.
        siglip_processor_id: If set, emit SigLIP2 inputs for the first image.
        clip_image_size: Fallback ``images_clip`` resolution.
        tuna_token_ids: Dict of special token ids from the model wrapper.
        tar_filename_format: Format string for TAR filenames.
    """

    def __init__(
        self,
        annotation_dir: str,
        image_root: str,
        tokenizer,
        image_size: int | tuple[int, int] = 512,
        max_text_length: int = 4096,
        data_filename_format: str = "data_part_{}.jsonl",
        range_start: int = 1,
        range_end: int = 2,
        sample_num: int | None = None,
        center_crop: bool = True,
        num_image_tokens: int | None = None,
        siglip_processor_id: str | None = None,
        clip_image_size: int | tuple[int, int] = 384,
        tuna_token_ids: dict[str, int] | None = None,
        tar_filename_format: str = "batch_{}.tar",
        **kwargs,
    ) -> None:
        super().__init__()
        self.annotation_dir = annotation_dir
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_size = (
            (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        )
        self.max_text_length = max_text_length
        self.data_filename_format = data_filename_format
        self.range_start = range_start
        self.range_end = range_end
        self.tar_filename_format = tar_filename_format

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
            raise ValueError("tuna_token_ids is required for VLMDataset")

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

        # Load records and build TAR index.
        self.records: list[dict[str, Any]] = self._load_records()
        self._tar_index: dict[str, tuple[str, str]] = {}  # relative_path -> (tar_path, member_name)
        self._build_tar_index()

        logger.info(
            f"VLMDataset: loaded {len(self.records)} records from {annotation_dir} "
            f"(parts {range_start}-{range_end - 1}, image_size={self.image_size})"
        )

    # ---- record loading ----------------------------------------------------

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for part in range(self.range_start, self.range_end):
            fname = self.data_filename_format.format(part)
            fpath = os.path.join(self.annotation_dir, fname)
            if not os.path.isfile(fpath):
                logger.warning(f"Annotation file not found: {fpath}")
                continue
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Skip malformed JSON in {fpath}")
        return records

    # ---- TAR index ---------------------------------------------------------

    def _build_tar_index(self) -> None:
        """Scan TAR files and map every member name to (tar_path, member_name)."""
        tar_files = sorted(
            f for f in os.listdir(self.image_root)
            if f.endswith(".tar")
        )
        for tar_name in tar_files:
            tar_path = os.path.join(self.image_root, tar_name)
            try:
                with tarfile.open(tar_path, "r") as tar:
                    for member in tar.getmembers():
                        if member.isfile():
                            self._tar_index[member.name] = (tar_path, member.name)
            except tarfile.TarError as e:
                logger.warning(f"Failed to read TAR {tar_path}: {e}")

    @staticmethod
    @lru_cache(maxsize=16)
    def _open_tar(tar_path: str) -> tarfile.TarFile:
        return tarfile.open(tar_path, "r")

    def _load_image_from_tar(self, relative_path: str) -> Image.Image:
        entry = self._tar_index.get(relative_path)
        if entry is None:
            raise FileNotFoundError(
                f"Image '{relative_path}' not found in any TAR under {self.image_root}"
            )
        tar_path, member_name = entry
        tar = self._open_tar(tar_path)
        data = tar.extractfile(member_name)
        if data is None:
            raise RuntimeError(f"Failed to extract {member_name} from {tar_path}")
        img = Image.open(io.BytesIO(data.read()))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    # ---- conversation parsing ----------------------------------------------

    def _parse_turn(self, turn: dict[str, Any]) -> tuple[str, list[str], str]:
        """Parse one conversation turn.

        Returns:
            ``(role, image_paths, text_string)``.
        """
        role = turn.get("role", "user")
        content_list = turn.get("content", [])
        image_paths: list[str] = []
        text_parts: list[str] = []

        for item in content_list:
            if item.get("type") == "image":
                img_info = item.get("image", {})
                rel_path = img_info.get("relative_path", "")
                if rel_path:
                    image_paths.append(rel_path)
            elif item.get("type") == "text":
                text_info = item.get("text", {})
                text_str = text_info.get("string", "")
                if text_str:
                    text_parts.append(text_str)

        return role, image_paths, " ".join(text_parts)

    def _extract_conversation(self, record: dict[str, Any]) -> dict[str, Any]:
        """Extract structured conversation from a record.

        Returns a dict with:
          * ``image_paths``: all unique image paths referenced
          * ``user_texts``: list of user text strings (per turn)
          * ``assistant_texts``: list of assistant text strings (per turn)
          * ``messages``: HF chat-template messages list
          * ``sentence``: last assistant response (for logging)
        """
        data = record.get("data", [])
        image_paths: list[str] = []
        user_texts: list[str] = []
        assistant_texts: list[str] = []
        messages: list[dict[str, Any]] = []

        for turn in data:
            role, imgs, text = self._parse_turn(turn)
            image_paths.extend(imgs)

            if role == "assistant":
                assistant_texts.append(text)
                messages.append({"role": "assistant", "content": text})
            else:
                user_texts.append(text)
                # Build content with image placeholders + text.
                content: list[dict[str, Any]] = []
                for img_path in imgs:
                    content.append({"type": "image"})
                if text:
                    content.append({"type": "text", "text": text})
                messages.append({"role": "user", "content": content})

        sentence = assistant_texts[-1] if assistant_texts else ""

        return {
            "image_paths": image_paths,
            "user_texts": user_texts,
            "assistant_texts": assistant_texts,
            "messages": messages,
            "sentence": sentence,
        }

    # ---- main --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        conv = self._extract_conversation(record)

        # 1. Load images.
        image_pils: list[Image.Image] = []
        for img_path in conv["image_paths"]:
            try:
                image_pils.append(self._load_image_from_tar(img_path))
            except (FileNotFoundError, RuntimeError) as e:
                logger.warning(f"Skip image {img_path}: {e}")
                # Create a dummy black image as fallback.
                image_pils.append(
                    Image.new("RGB", (self.image_size[1], self.image_size[0]))
                )

        # 2. Build chat template token sequence.
        # Inject <image> placeholder tokens so the model knows where images go.
        try:
            token_ids = self.tokenizer.apply_chat_template(
                conv["messages"],
                add_generation_prompt=False,
                tokenize=True,
            )
        except Exception:
            # Fallback: use the last assistant text as response, first user text as prompt.
            response = conv["sentence"]
            prompt = conv["user_texts"][0] if conv["user_texts"] else ""
            token_ids = self.tokenizer(
                prompt + " " + response, add_special_tokens=False,
                truncation=True, max_length=self.max_text_length,
            )["input_ids"]

        text_token_list = list(token_ids)

        # 3. Build prompt and response split.
        # Find <image> placeholders and prepare prompt_tokens / response_tokens.
        all_eos_ids = [i for i, tid in enumerate(text_token_list) if tid == self.eos_id]

        if len(all_eos_ids) >= 2:
            # Chat template produces: sys_eos, user_eos, assistant_eos, ...
            # Assistant turn starts after the second-to-last eos.
            assistant_start = all_eos_ids[-2] + 1
            prompt_tokens = text_token_list[:assistant_start]
            response_tokens = text_token_list[assistant_start:]
        else:
            # Simple case: last 50% as response.
            split = len(text_token_list) // 2
            prompt_tokens = text_token_list[:split]
            response_tokens = text_token_list[split:]

        # 4. Find <image> token positions and replace with image spans.
        # Count images in the prompt portion to determine num_image_tokens total.
        n_images = len(image_pils)
        if n_images == 0:
            n_images = 1  # at least one slot

        per_image_tokens = self.num_image_tokens // max(n_images, 1)
        total_img_tokens = per_image_tokens * n_images

        # Process prompt_tokens to replace <image> placeholders if they exist.
        img_positions = [
            i for i, tid in enumerate(prompt_tokens)
            if tid == self.img_id
        ]
        if img_positions:
            # Replace each <image> with [boi][img_pad]*P[eoi].
            expanded: list[int] = []
            last_idx = 0
            for img_idx in img_positions:
                expanded.extend(prompt_tokens[last_idx:img_idx])
                expanded.append(self.boi_id)
                expanded.extend([self.img_pad_id] * per_image_tokens)
                expanded.append(self.eoi_id)
                last_idx = img_idx + 1
            expanded.extend(prompt_tokens[last_idx:])
            prompt_tokens = expanded

        # 5. Format as MMU sequence.
        tt, tl, mp, tm, im = format_sequence_und(
            text_tokens=response_tokens,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            boi_id=self.boi_id,
            eoi_id=self.eoi_id,
            pad_id=self.pad_id,
            img_pad_id=self.img_pad_id,
            num_image_tokens=total_img_tokens,
            max_seq_len=self.max_text_length,
            prompt_tokens=prompt_tokens,
        )

        # 6. Stack images.
        image_tensors = [self.image_transform(p) for p in image_pils]
        if len(image_tensors) == 1:
            images = image_tensors[0]
        else:
            images = torch.stack(image_tensors)

        out: dict[str, Any] = {
            "images": images,
            "text_tokens": tt.long(),
            "text_labels": tl.long(),
            "text_masks": tm.bool(),
            "image_masks": im.bool(),
            "modality_positions": mp.long(),
            "data_type": "mmu",
            "sentence": conv["sentence"],
        }

        # 7. Build images_clip from the first image.
        first_pil = image_pils[0] if image_pils else None
        if first_pil is not None:
            if self.siglip_transform is not None:
                sig = self.siglip_transform(first_pil)
                out["images_clip"] = sig["pixel_values"]
                out["siglip_pixel_attention_mask"] = sig["pixel_attention_mask"]
                out["siglip_spatial_shapes"] = sig["spatial_shapes"]
            else:
                out["images_clip"] = self.clip_image_transform(first_pil)
        else:
            out["images_clip"] = torch.zeros(
                3, self.clip_image_size[0], self.clip_image_size[1]
            )

        return out
