"""
M2 — Counterfactual Text-Guided U-Net
=====================================

Ablation model: M2

Components
----------
Text conditioning:              True
Prompt-image compatibility:     False
Counterfactual prompt training: True
Boundary/uncertainty refinement:False

Scientific protocol
-------------------
- Reuses exactly the same train/validation/test CSV files.
- Uses the same augmentation, weighted sampler, optimizer, scheduler, epochs,
  early stopping, TTA, and metric definitions as the other M0-M6 scripts.
- Selects ONE deployment threshold from the validation set only.
- Applies that frozen threshold once to all test prompt modes.
- Saves per-image results, run summary, configuration, history, and weights.
- Appends the run summary to:
      ./ablation_results_multisource/all_runs.csv

Required companion file
-----------------------
Place this script beside:
    train_text_guided_unet.py
    multisource_split_helper.py

Run examples
------------
    python m2_counterfactual_text_unet.py
    python m2_counterfactual_text_unet.py --seed 123
    python m2_counterfactual_text_unet.py --seed 2026 --epochs 120

This script is intentionally complete and independently runnable.
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from scipy.ndimage import (
    convolve,
    distance_transform_edt,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

import train_text_guided_unet as base
from multisource_split_helper import (
    combine_kvasir_and_clinicdb,
)


# ============================================================
# FIXED EXPERIMENT DEFINITION
# ============================================================
EXPERIMENT_ID = "M2"
EXPERIMENT_NAME = "counterfactual_text_unet"

USE_TEXT_CONDITIONING = True
USE_COMPATIBILITY_BRANCH = False
USE_COUNTERFACTUAL_TRAINING = True
USE_BOUNDARY_REFINEMENT = False


# ============================================================
# DATA AND TRAINING CONFIGURATION
# Edit only paths when necessary. Keep all other settings equal
# across M0-M6 for a fair ablation.
# ============================================================
CSV_PATH = "./outputs/metadata_50.csv"

# Kvasir-SEG source paths. Edit when your folders differ.
IMAGES_DIR = r"E:\col\kvasir-seg\t4sdeep\LLM\Kvasir-SEG/images"
MASKS_DIR = r"E:\col\kvasir-seg\t4sdeep\LLM\Kvasir-SEG/masks"

# CVC-ClinicDB is included in training and validation for this
# multi-source ablation protocol. It remains excluded from the test set.
CLINICDB_METADATA_CSV = "./outputscvc/metadata_cvc_clinicdb.csv"
CLINICDB_IMAGES_DIR = r"E:\col\kvasir-seg\t4sdeep\LLM\CVC-ClinicDB\images"
CLINICDB_MASKS_DIR = r"E:\col\kvasir-seg\t4sdeep\LLM\CVC-ClinicDB/masks"
CLINICDB_SPLIT_DIR = "./multisource_splits2"
CLINICDB_VALIDATION_FRACTION = 0.15

BASELINE_SPLIT_DIR = "./unet_outputs"
MASTER_OUTPUT_DIR = "./ablation_results_multisource2"

TEXT_MODEL_NAME = "distilbert/distilbert-base-uncased"
TEXT_CACHE_PATH = (
    "./text_guided_unet_outputs/"
    "prompt_embedding_cache.pt"
)



IMAGE_SIZE = 352
DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_EPOCHS = 120
DEFAULT_NUM_WORKERS = 0

LEARNING_RATE = 2e-4
MIN_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 32
EARLY_STOPPING_PATIENCE = 25

USE_AMP = True
USE_WEIGHTED_SAMPLER = True
USE_TEST_TIME_AUGMENTATION = True

GENERIC_PROMPT_PROBABILITY = 0.20
GENERIC_COMPATIBILITY_TARGET = 0.50

COARSE_LOSS_WEIGHT = 0.25
BOUNDARY_LOSS_WEIGHT = 0.20
WRONG_PROMPT_SEG_LOSS_WEIGHT = 0.35
COMPATIBILITY_LOSS_WEIGHT = 0.15
COUNTERFACTUAL_CONSISTENCY_WEIGHT = 0.10
COMPATIBILITY_MARGIN_WEIGHT = 0.05
COMPATIBILITY_MARGIN = 0.30

BOUNDARY_KERNEL_SIZE = 5
BOUNDARY_TOLERANCE_PIXELS = 2

# Additional foreground-map evaluation metrics.
# S-measure combines object-aware and region-aware terms with alpha=0.5.
S_MEASURE_ALPHA = 0.50

# Standard foreground-map evaluation uses beta^2 = 0.3.
WEIGHTED_F_BETA_SQUARED = 0.30
WEIGHTED_F_GAUSSIAN_KERNEL_SIZE = 7
WEIGHTED_F_GAUSSIAN_SIGMA = 5.0

MAX_ROC_AP_PIXELS = 2_000_000
FIXED_PIXEL_SAMPLE_SEED = 314159

GENERIC_PROMPT = base.GENERIC_PROMPT


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Train and evaluate {EXPERIMENT_ID} "
            f"({EXPERIMENT_NAME})."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed. Recommended: 42, 123, 2026.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable horizontal-flip test-time augmentation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Enable stricter CUDA determinism. "
            "This may reduce training speed."
        ),
    )
    parser.add_argument(
        "--clinicdb-metadata",
        type=str,
        default=CLINICDB_METADATA_CSV,
        help="Metadata CSV created by prepare_clinicdb_metadata.py.",
    )
    parser.add_argument(
        "--clinicdb-images-dir",
        type=str,
        default=CLINICDB_IMAGES_DIR,
    )
    parser.add_argument(
        "--clinicdb-masks-dir",
        type=str,
        default=CLINICDB_MASKS_DIR,
    )
    parser.add_argument(
        "--clinicdb-split-dir",
        type=str,
        default=CLINICDB_SPLIT_DIR,
    )
    parser.add_argument(
        "--clinicdb-val-fraction",
        type=float,
        default=CLINICDB_VALIDATION_FRACTION,
        help="Fixed ClinicDB validation fraction; default 0.15.",
    )

    return parser.parse_args()


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================
# PATH AND CHECKPOINT HELPERS
# ============================================================
def safe_torch_load(
    path: str,
    device: torch.device,
):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def save_json(
    path: str,
    payload: dict,
) -> None:
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


# ============================================================
# STRONG COUNTERFACTUAL PROMPT SELECTION
# ============================================================
def build_wrong_prompt_indices(
    dataframe: pd.DataFrame,
    seed: int,
) -> List[int]:
    """
    Select a deliberately mismatched prompt with this priority:
      1. different count category,
      2. different size category,
      3. different location category,
      4. any different image.

    The same procedure is used in every text-based ablation.
    """
    df = dataframe.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    indices: List[int] = []

    for index, row in df.iterrows():
        candidates = df[
            df["count_cat"] != row["count_cat"]
        ].index.to_numpy()

        if len(candidates) == 0:
            candidates = df[
                df["size_cat"] != row["size_cat"]
            ].index.to_numpy()

        if (
            len(candidates) == 0
            and "location_cat" in df.columns
        ):
            candidates = df[
                df["location_cat"]
                != row.get("location_cat", "none")
            ].index.to_numpy()

        if len(candidates) == 0:
            candidates = df.index[
                df.index != index
            ].to_numpy()

        chosen = (
            int(rng.choice(candidates))
            if len(candidates) > 0
            else index
        )
        indices.append(chosen)

    return indices


# ============================================================
# DATASETS
# ============================================================
class AblationTrainingDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        prompt_embeddings: Dict[str, torch.Tensor],
        images_root: str,
        masks_root: str,
        image_size: int,
        augment: bool,
        generic_prompt_probability: float,
        seed: int,
        text_dim: int,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.prompt_embeddings = prompt_embeddings
        self.images_root = images_root
        self.masks_root = masks_root
        self.image_size = image_size
        self.augment = augment
        self.generic_prompt_probability = (
            generic_prompt_probability
        )
        self.text_dim = text_dim
        self.wrong_prompt_indices = (
            build_wrong_prompt_indices(
                self.df,
                seed,
            )
        )

    def __len__(self) -> int:
        return len(self.df)

    def _augment_pair(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.50:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        if random.random() < 0.30:
            image = np.flipud(image)
            mask = np.flipud(mask)

        if random.random() < 0.70:
            image, mask = base.random_affine(
                image,
                mask,
            )

        image = base.color_augmentation(image)

        return (
            np.ascontiguousarray(image),
            np.ascontiguousarray(mask),
        )

    def _embedding(
        self,
        prompt: str,
    ) -> torch.Tensor:
        if not USE_TEXT_CONDITIONING:
            return torch.zeros(
                self.text_dim,
                dtype=torch.float32,
            )

        return self.prompt_embeddings[
            prompt
        ].clone().float()

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        image_path = base.resolve_path(
            row["image_path"],
            self.images_root,
        )
        mask_path = base.resolve_path(
            row["mask_path"],
            self.masks_root,
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = (mask > 0).astype(np.uint8)

        if self.augment:
            image, mask = self._augment_pair(
                image,
                mask,
            )

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = np.expand_dims(
            (mask > 0).astype(np.float32),
            axis=0,
        )

        oracle_prompt = str(row["oracle_prompt"])

        if (
            USE_TEXT_CONDITIONING
            and random.random()
            < self.generic_prompt_probability
        ):
            primary_prompt = GENERIC_PROMPT
            primary_mode = "generic"
            primary_compatibility_target = (
                GENERIC_COMPATIBILITY_TARGET
            )
        else:
            primary_prompt = oracle_prompt
            primary_mode = (
                "oracle"
                if USE_TEXT_CONDITIONING
                else "image_only"
            )
            primary_compatibility_target = 1.0

        wrong_row = self.df.iloc[
            self.wrong_prompt_indices[index]
        ]
        wrong_prompt = str(
            wrong_row["oracle_prompt"]
        )

        return (
            torch.from_numpy(
                np.ascontiguousarray(image)
            ),
            torch.from_numpy(
                np.ascontiguousarray(mask)
            ),
            self._embedding(primary_prompt),
            self._embedding(wrong_prompt),
            torch.tensor(
                primary_compatibility_target,
                dtype=torch.float32,
            ),
            base.filename_from_csv_path(
                row["image_path"]
            ),
            str(row.get("count_cat", "unknown")),
            str(row.get("size_cat", "unknown")),
            str(row.get("shape_cat", "unknown")),
            primary_mode,
            primary_prompt,
            wrong_prompt,
        )


class AblationEvaluationDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        prompt_embeddings: Dict[str, torch.Tensor],
        images_root: str,
        masks_root: str,
        image_size: int,
        prompt_mode: str,
        seed: int,
        text_dim: int,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.prompt_embeddings = prompt_embeddings
        self.images_root = images_root
        self.masks_root = masks_root
        self.image_size = image_size
        self.prompt_mode = prompt_mode
        self.text_dim = text_dim

        allowed = (
            {"image_only"}
            if not USE_TEXT_CONDITIONING
            else {"oracle", "generic", "wrong"}
        )

        if prompt_mode not in allowed:
            raise ValueError(
                f"Invalid prompt mode {prompt_mode}; "
                f"expected one of {sorted(allowed)}"
            )

        self.wrong_prompt_indices = (
            build_wrong_prompt_indices(
                self.df,
                seed,
            )
        )

    def __len__(self) -> int:
        return len(self.df)

    def _embedding(
        self,
        prompt: str,
    ) -> torch.Tensor:
        if not USE_TEXT_CONDITIONING:
            return torch.zeros(
                self.text_dim,
                dtype=torch.float32,
            )

        return self.prompt_embeddings[
            prompt
        ].clone().float()

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        image_path = base.resolve_path(
            row["image_path"],
            self.images_root,
        )
        mask_path = base.resolve_path(
            row["mask_path"],
            self.masks_root,
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = np.expand_dims(
            (mask > 0).astype(np.float32),
            axis=0,
        )

        if self.prompt_mode == "image_only":
            prompt = GENERIC_PROMPT
            compatibility_target = 1.0
        elif self.prompt_mode == "oracle":
            prompt = str(row["oracle_prompt"])
            compatibility_target = 1.0
        elif self.prompt_mode == "generic":
            prompt = GENERIC_PROMPT
            compatibility_target = (
                GENERIC_COMPATIBILITY_TARGET
            )
        else:
            wrong_row = self.df.iloc[
                self.wrong_prompt_indices[index]
            ]
            prompt = str(
                wrong_row["oracle_prompt"]
            )
            compatibility_target = 0.0

        return (
            torch.from_numpy(
                np.ascontiguousarray(image)
            ),
            torch.from_numpy(
                np.ascontiguousarray(mask)
            ),
            self._embedding(prompt),
            torch.tensor(
                compatibility_target,
                dtype=torch.float32,
            ),
            base.filename_from_csv_path(
                row["image_path"]
            ),
            str(row.get("count_cat", "unknown")),
            str(row.get("size_cat", "unknown")),
            str(row.get("shape_cat", "unknown")),
            prompt,
        )


# ============================================================
# MODEL COMPONENTS
# ============================================================
class CompatibilityFiLM(nn.Module):
    """
    Compatibility-gated FiLM:

        F_tilde = F * (1 + c * gamma(T)) + c * beta(T)

    c is an auxiliary prompt-image compatibility conditioning signal.
    It is not assumed to be a calibrated probability.
    """

    def __init__(
        self,
        text_dim: int,
        feature_channels: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                feature_channels * 2,
            ),
        )

        final_layer = self.projection[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        feature: torch.Tensor,
        text_embedding: torch.Tensor,
        compatibility: torch.Tensor,
    ) -> torch.Tensor:
        gamma_beta = self.projection(
            text_embedding
        )
        gamma, beta = gamma_beta.chunk(
            2,
            dim=1,
        )

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        gate = compatibility.unsqueeze(-1).unsqueeze(-1)

        return (
            feature * (1.0 + gate * gamma)
            + gate * beta
        )


class PromptImageCompatibilityBranch(nn.Module):
    """
    Auxiliary image-prompt compatibility branch.

    The output is used to condition FiLM. It should not be described
    as a calibrated confidence score unless calibration is demonstrated.
    """

    def __init__(
        self,
        image_channels: int,
        text_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.image_projection = nn.Sequential(
            nn.Linear(
                image_channels,
                hidden_dim,
            ),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.text_projection = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(
                text_dim,
                hidden_dim,
            ),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                hidden_dim * 4,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        bottleneck: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        image_vector = F.adaptive_avg_pool2d(
            bottleneck,
            output_size=1,
        ).flatten(1)

        image_vector = self.image_projection(
            image_vector
        )
        text_vector = self.text_projection(
            text_embedding
        )

        descriptor = torch.cat(
            [
                image_vector,
                text_vector,
                torch.abs(
                    image_vector - text_vector
                ),
                image_vector * text_vector,
            ],
            dim=1,
        )

        return self.classifier(
            descriptor
        ).squeeze(1)


class AblationUNet(nn.Module):
    def __init__(
        self,
        text_dim: int,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        b = base_channels
        self.pool = nn.MaxPool2d(2)

        self.enc1 = base.DoubleConv(
            in_channels,
            b,
        )
        self.enc2 = base.DoubleConv(
            b,
            b * 2,
        )
        self.enc3 = base.DoubleConv(
            b * 2,
            b * 4,
        )
        self.enc4 = base.DoubleConv(
            b * 4,
            b * 8,
            dropout=0.10,
        )

        self.bottleneck = base.DoubleConv(
            b * 8,
            b * 16,
            dropout=0.20,
        )

        if USE_TEXT_CONDITIONING:
            if USE_COMPATIBILITY_BRANCH:
                self.compatibility_branch = (
                    PromptImageCompatibilityBranch(
                        image_channels=b * 16,
                        text_dim=text_dim,
                    )
                )
                film_class = CompatibilityFiLM
            else:
                self.compatibility_branch = None
                film_class = base.FiLMConditioning

            self.bottleneck_film = film_class(
                text_dim,
                b * 16,
            )
        else:
            self.compatibility_branch = None
            self.bottleneck_film = None

        self.up4 = nn.ConvTranspose2d(
            b * 16,
            b * 8,
            kernel_size=2,
            stride=2,
        )
        self.dec4 = base.DoubleConv(
            b * 16,
            b * 8,
            dropout=0.10,
        )

        self.up3 = nn.ConvTranspose2d(
            b * 8,
            b * 4,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = base.DoubleConv(
            b * 8,
            b * 4,
        )

        self.up2 = nn.ConvTranspose2d(
            b * 4,
            b * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = base.DoubleConv(
            b * 4,
            b * 2,
        )

        self.up1 = nn.ConvTranspose2d(
            b * 2,
            b,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = base.DoubleConv(
            b * 2,
            b,
        )

        if USE_TEXT_CONDITIONING:
            film_class = (
                CompatibilityFiLM
                if USE_COMPATIBILITY_BRANCH
                else base.FiLMConditioning
            )

            self.dec4_film = film_class(
                text_dim,
                b * 8,
            )
            self.dec3_film = film_class(
                text_dim,
                b * 4,
            )
            self.dec2_film = film_class(
                text_dim,
                b * 2,
            )
            self.dec1_film = film_class(
                text_dim,
                b,
            )
        else:
            self.dec4_film = None
            self.dec3_film = None
            self.dec2_film = None
            self.dec1_film = None

        self.coarse_head = nn.Conv2d(
            b,
            out_channels,
            kernel_size=1,
        )

        if USE_BOUNDARY_REFINEMENT:
            self.boundary_head = nn.Sequential(
                nn.Conv2d(
                    b,
                    b,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    b,
                    1,
                    kernel_size=1,
                ),
            )

            self.refinement_head = nn.Sequential(
                nn.Conv2d(
                    b + 3,
                    b,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(b),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    b,
                    b // 2,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(b // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    b // 2,
                    out_channels,
                    kernel_size=1,
                ),
            )
        else:
            self.boundary_head = None
            self.refinement_head = None

    def _condition(
        self,
        feature: torch.Tensor,
        text_embedding: torch.Tensor,
        compatibility: torch.Tensor,
        film_module: Optional[nn.Module],
    ) -> torch.Tensor:
        if not USE_TEXT_CONDITIONING:
            return feature

        if USE_COMPATIBILITY_BRANCH:
            return film_module(
                feature,
                text_embedding,
                compatibility,
            )

        return film_module(
            feature,
            text_embedding,
        )

    def forward(
        self,
        image: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        x = self.bottleneck(
            self.pool(e4)
        )

        batch_size = image.shape[0]

        if USE_COMPATIBILITY_BRANCH:
            compatibility_logit = (
                self.compatibility_branch(
                    x,
                    text_embedding,
                )
            )
            compatibility = torch.sigmoid(
                compatibility_logit
            ).unsqueeze(1)
        else:
            compatibility_logit = torch.zeros(
                batch_size,
                device=image.device,
                dtype=image.dtype,
            )
            compatibility = torch.ones(
                batch_size,
                1,
                device=image.device,
                dtype=image.dtype,
            )

        x = self._condition(
            x,
            text_embedding,
            compatibility,
            self.bottleneck_film,
        )

        x = self.up4(x)
        x = torch.cat([x, e4], dim=1)
        x = self.dec4(x)
        x = self._condition(
            x,
            text_embedding,
            compatibility,
            self.dec4_film,
        )

        x = self.up3(x)
        x = torch.cat([x, e3], dim=1)
        x = self.dec3(x)
        x = self._condition(
            x,
            text_embedding,
            compatibility,
            self.dec3_film,
        )

        x = self.up2(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)
        x = self._condition(
            x,
            text_embedding,
            compatibility,
            self.dec2_film,
        )

        x = self.up1(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)
        x = self._condition(
            x,
            text_embedding,
            compatibility,
            self.dec1_film,
        )

        coarse_logits = self.coarse_head(x)

        if USE_BOUNDARY_REFINEMENT:
            boundary_logits = self.boundary_head(x)

            coarse_probability = torch.sigmoid(
                coarse_logits
            )
            boundary_probability = torch.sigmoid(
                boundary_logits
            )

            uncertainty = (
                1.0
                - torch.abs(
                    2.0 * coarse_probability
                    - 1.0
                )
            )

            refinement_input = torch.cat(
                [
                    x,
                    coarse_probability,
                    boundary_probability,
                    uncertainty,
                ],
                dim=1,
            )

            residual_logits = self.refinement_head(
                refinement_input
            )

            final_logits = (
                coarse_logits
                + residual_logits
                * (0.5 + uncertainty)
            )
        else:
            boundary_logits = torch.zeros_like(
                coarse_logits
            )
            uncertainty = torch.zeros_like(
                coarse_logits
            )
            final_logits = coarse_logits

        return {
            "final_logits": final_logits,
            "coarse_logits": coarse_logits,
            "boundary_logits": boundary_logits,
            "uncertainty": uncertainty,
            "compatibility_logit": (
                compatibility_logit
            ),
            "compatibility_score": (
                compatibility.squeeze(1)
            ),
        }


# ============================================================
# LOSSES
# ============================================================
bce_logits = nn.BCEWithLogitsLoss()


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)

    probability = probability.flatten(
        start_dim=1
    )
    targets = targets.flatten(
        start_dim=1
    )

    intersection = (
        probability * targets
    ).sum(dim=1)

    denominator = (
        probability.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return (
        0.40 * bce_logits(
            logits,
            targets,
        )
        + 0.60 * soft_dice_loss(
            logits,
            targets,
        )
    )


def make_boundary_target(
    mask: torch.Tensor,
    kernel_size: int = BOUNDARY_KERNEL_SIZE,
) -> torch.Tensor:
    padding = kernel_size // 2

    dilated = F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    eroded = -F.max_pool2d(
        -mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    return (
        dilated - eroded
    ).clamp(0.0, 1.0)


def boundary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return (
        0.60 * bce_logits(
            logits,
            targets,
        )
        + 0.40 * soft_dice_loss(
            logits,
            targets,
        )
    )


def compute_training_loss(
    primary_output: Dict[str, torch.Tensor],
    wrong_output: Optional[
        Dict[str, torch.Tensor]
    ],
    masks: torch.Tensor,
    primary_compatibility_target: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    primary_seg = segmentation_loss(
        primary_output["final_logits"],
        masks,
    )

    total = primary_seg

    zero = torch.zeros(
        (),
        device=masks.device,
    )

    coarse = zero
    boundary = zero
    wrong_seg = zero
    compatibility_loss = zero
    consistency = zero
    margin = zero

    if USE_BOUNDARY_REFINEMENT:
        coarse = segmentation_loss(
            primary_output["coarse_logits"],
            masks,
        )

        boundary_target = make_boundary_target(
            masks
        )

        boundary = boundary_loss(
            primary_output["boundary_logits"],
            boundary_target,
        )

        total = (
            total
            + COARSE_LOSS_WEIGHT * coarse
            + BOUNDARY_LOSS_WEIGHT * boundary
        )

    if USE_COUNTERFACTUAL_TRAINING:
        if wrong_output is None:
            raise RuntimeError(
                "Counterfactual training requires "
                "a wrong-prompt forward pass."
            )

        wrong_seg = segmentation_loss(
            wrong_output["final_logits"],
            masks,
        )

        primary_probability = torch.sigmoid(
            primary_output["final_logits"]
        ).detach()

        wrong_probability = torch.sigmoid(
            wrong_output["final_logits"]
        )

        consistency = F.smooth_l1_loss(
            wrong_probability,
            primary_probability,
        )

        total = (
            total
            + WRONG_PROMPT_SEG_LOSS_WEIGHT
            * wrong_seg
            + COUNTERFACTUAL_CONSISTENCY_WEIGHT
            * consistency
        )

    if USE_COMPATIBILITY_BRANCH:
        if wrong_output is None:
            raise RuntimeError(
                "Compatibility supervision requires "
                "a wrong-prompt forward pass."
            )

        primary_comp = (
            F.binary_cross_entropy_with_logits(
                primary_output[
                    "compatibility_logit"
                ],
                primary_compatibility_target,
            )
        )

        wrong_target = torch.zeros_like(
            primary_compatibility_target
        )

        wrong_comp = (
            F.binary_cross_entropy_with_logits(
                wrong_output[
                    "compatibility_logit"
                ],
                wrong_target,
            )
        )

        compatibility_loss = 0.5 * (
            primary_comp + wrong_comp
        )

        margin = F.relu(
            COMPATIBILITY_MARGIN
            - (
                primary_output[
                    "compatibility_score"
                ]
                - wrong_output[
                    "compatibility_score"
                ]
            )
        ).mean()

        total = (
            total
            + COMPATIBILITY_LOSS_WEIGHT
            * compatibility_loss
            + COMPATIBILITY_MARGIN_WEIGHT
            * margin
        )

    components = {
        "primary_seg": float(
            primary_seg.detach()
        ),
        "coarse": float(coarse.detach()),
        "boundary": float(boundary.detach()),
        "wrong_seg": float(
            wrong_seg.detach()
        ),
        "compatibility": float(
            compatibility_loss.detach()
        ),
        "consistency": float(
            consistency.detach()
        ),
        "margin": float(margin.detach()),
    }

    return total, components


# ============================================================
# METRICS
# ============================================================
def batch_dice_iou(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.flatten(
        start_dim=1
    )
    targets = targets.flatten(
        start_dim=1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    prediction_sum = predictions.sum(dim=1)
    target_sum = targets.sum(dim=1)

    union = (
        prediction_sum
        + target_sum
        - intersection
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        prediction_sum
        + target_sum
        + eps
    )

    iou = (
        intersection + eps
    ) / (
        union + eps
    )

    return dice, iou


def per_image_binary_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    eps: float = 1e-7,
) -> Dict[str, np.ndarray]:
    predictions = (
        probabilities >= threshold
    ).float()

    prediction_flat = predictions.flatten(
        start_dim=1
    )
    target_flat = targets.flatten(
        start_dim=1
    )
    probability_flat = probabilities.flatten(
        start_dim=1
    )

    tp = (
        prediction_flat * target_flat
    ).sum(dim=1)

    fp = (
        prediction_flat
        * (1.0 - target_flat)
    ).sum(dim=1)

    fn = (
        (1.0 - prediction_flat)
        * target_flat
    ).sum(dim=1)

    tn = (
        (1.0 - prediction_flat)
        * (1.0 - target_flat)
    ).sum(dim=1)

    dice = (
        2.0 * tp + eps
    ) / (
        2.0 * tp + fp + fn + eps
    )

    iou = (
        tp + eps
    ) / (
        tp + fp + fn + eps
    )

    precision = (
        tp + eps
    ) / (
        tp + fp + eps
    )

    recall = (
        tp + eps
    ) / (
        tp + fn + eps
    )

    specificity = (
        tn + eps
    ) / (
        tn + fp + eps
    )

    mae = torch.abs(
        probability_flat - target_flat
    ).mean(dim=1)

    return {
        "dice": dice.cpu().numpy(),
        "iou": iou.cpu().numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "specificity": specificity.cpu().numpy(),
        "mae": mae.cpu().numpy(),
    }



def _gaussian_kernel(
    kernel_size: int,
    sigma: float,
) -> np.ndarray:
    """Create the normalized Gaussian kernel used by weighted F-measure."""
    radius = kernel_size // 2
    coordinates = np.arange(
        -radius,
        radius + 1,
        dtype=np.float64,
    )
    xx, yy = np.meshgrid(
        coordinates,
        coordinates,
    )

    kernel = np.exp(
        -(xx * xx + yy * yy)
        / (2.0 * sigma * sigma)
    )
    kernel /= kernel.sum()

    return kernel


WEIGHTED_F_GAUSSIAN_KERNEL = _gaussian_kernel(
    WEIGHTED_F_GAUSSIAN_KERNEL_SIZE,
    WEIGHTED_F_GAUSSIAN_SIGMA,
)


def weighted_f_measure_single(
    probability: np.ndarray,
    target: np.ndarray,
    beta_squared: float = WEIGHTED_F_BETA_SQUARED,
    eps: float = 1e-7,
) -> float:
    """
    Weighted F-measure F_beta^omega for one soft prediction map.

    The spatial weighting follows the foreground-map evaluation procedure:
    errors are propagated to the nearest foreground pixel, smoothed with a
    Gaussian kernel, and background errors are weighted by their distance
    from the foreground.
    """
    probability = np.clip(
        probability.astype(np.float64),
        0.0,
        1.0,
    )
    target_bool = target.astype(bool)
    target_float = target_bool.astype(np.float64)

    # The standard weighted-F protocol returns zero for an empty foreground.
    # Polyp datasets normally contain foreground in every ground-truth mask.
    if not target_bool.any():
        return 0.0

    error = np.abs(
        probability - target_float
    )
    background = ~target_bool

    if background.any():
        distance, nearest_indices = (
            distance_transform_edt(
                background,
                return_indices=True,
            )
        )

        propagated_error = error.copy()
        propagated_error[background] = error[
            nearest_indices[0][background],
            nearest_indices[1][background],
        ]
    else:
        distance = np.zeros_like(
            probability,
            dtype=np.float64,
        )
        propagated_error = error.copy()

    smoothed_error = convolve(
        propagated_error,
        WEIGHTED_F_GAUSSIAN_KERNEL,
        mode="constant",
        cval=0.0,
    )

    minimum_error = error.copy()
    improved_foreground = (
        target_bool
        & (smoothed_error < error)
    )
    minimum_error[improved_foreground] = (
        smoothed_error[improved_foreground]
    )

    spatial_weight = np.ones_like(
        probability,
        dtype=np.float64,
    )

    if background.any():
        spatial_weight[background] = (
            2.0
            - np.exp(
                np.log(0.5)
                / 5.0
                * distance[background]
            )
        )

    weighted_error = (
        minimum_error * spatial_weight
    )

    weighted_true_positive = (
        float(target_bool.sum())
        - weighted_error[target_bool].sum()
    )
    weighted_false_positive = (
        weighted_error[background].sum()
        if background.any()
        else 0.0
    )

    weighted_recall = (
        1.0
        - weighted_error[
            target_bool
        ].mean()
    )
    weighted_precision = (
        weighted_true_positive
        / (
            weighted_true_positive
            + weighted_false_positive
            + eps
        )
    )

    score = (
        (1.0 + beta_squared)
        * weighted_precision
        * weighted_recall
        / (
            beta_squared
            * weighted_precision
            + weighted_recall
            + eps
        )
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )


def _s_object_score(
    values: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Object-aware similarity term used by S-measure."""
    if values.size == 0:
        return 0.0

    mean_value = float(values.mean())
    std_value = float(values.std(ddof=0))

    return float(
        2.0 * mean_value
        / (
            mean_value * mean_value
            + 1.0
            + 2.0 * std_value
            + eps
        )
    )


def _s_object_measure(
    probability: np.ndarray,
    target_bool: np.ndarray,
) -> float:
    foreground_weight = float(
        target_bool.mean()
    )

    foreground_values = probability[
        target_bool
    ]
    background_values = (
        1.0 - probability[~target_bool]
    )

    foreground_score = _s_object_score(
        foreground_values
    )
    background_score = _s_object_score(
        background_values
    )

    return float(
        foreground_weight
        * foreground_score
        + (1.0 - foreground_weight)
        * background_score
    )


def _s_centroid(
    target_bool: np.ndarray,
) -> Tuple[int, int]:
    """Return a safe (x, y) centroid for the four-region S-measure split."""
    height, width = target_bool.shape

    if not target_bool.any():
        center_x = width // 2
        center_y = height // 2
    else:
        y_coordinates, x_coordinates = np.nonzero(
            target_bool
        )
        center_x = int(
            np.round(x_coordinates.mean())
        )
        center_y = int(
            np.round(y_coordinates.mean())
        )

    if width > 1:
        center_x = int(
            np.clip(center_x, 1, width - 1)
        )
    else:
        center_x = 1

    if height > 1:
        center_y = int(
            np.clip(center_y, 1, height - 1)
        )
    else:
        center_y = 1

    return center_x, center_y


def _s_region_ssim(
    prediction_region: np.ndarray,
    target_region: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """SSIM-style region score used by the region-aware S-measure term."""
    if prediction_region.size == 0:
        return 0.0

    prediction_region = (
        prediction_region.astype(np.float64)
    )
    target_region = (
        target_region.astype(np.float64)
    )

    prediction_mean = float(
        prediction_region.mean()
    )
    target_mean = float(
        target_region.mean()
    )

    prediction_centered = (
        prediction_region - prediction_mean
    )
    target_centered = (
        target_region - target_mean
    )

    prediction_variance = float(
        np.mean(prediction_centered ** 2)
    )
    target_variance = float(
        np.mean(target_centered ** 2)
    )
    covariance = float(
        np.mean(
            prediction_centered
            * target_centered
        )
    )

    numerator = (
        4.0
        * prediction_mean
        * target_mean
        * covariance
    )
    denominator = (
        (
            prediction_mean ** 2
            + target_mean ** 2
        )
        * (
            prediction_variance
            + target_variance
        )
    )

    if abs(numerator) > eps:
        return float(
            numerator
            / (denominator + eps)
        )

    if abs(denominator) <= eps:
        return 1.0

    return 0.0


def _s_region_measure(
    probability: np.ndarray,
    target_bool: np.ndarray,
) -> float:
    height, width = target_bool.shape
    center_x, center_y = _s_centroid(
        target_bool
    )

    prediction_regions = [
        probability[:center_y, :center_x],
        probability[:center_y, center_x:],
        probability[center_y:, :center_x],
        probability[center_y:, center_x:],
    ]
    target_regions = [
        target_bool[:center_y, :center_x],
        target_bool[:center_y, center_x:],
        target_bool[center_y:, :center_x],
        target_bool[center_y:, center_x:],
    ]

    total_area = float(height * width)
    region_weights = [
        center_x * center_y / total_area,
        (width - center_x)
        * center_y
        / total_area,
        center_x
        * (height - center_y)
        / total_area,
        (width - center_x)
        * (height - center_y)
        / total_area,
    ]

    region_scores = [
        _s_region_ssim(
            prediction_region,
            target_region,
        )
        for prediction_region, target_region
        in zip(
            prediction_regions,
            target_regions,
        )
    ]

    return float(
        sum(
            weight * score
            for weight, score
            in zip(
                region_weights,
                region_scores,
            )
        )
    )


def s_measure_single(
    probability: np.ndarray,
    target: np.ndarray,
    alpha: float = S_MEASURE_ALPHA,
) -> float:
    """
    Structure-measure S_alpha for one soft prediction map.

    S_alpha = alpha * S_object + (1-alpha) * S_region.
    """
    probability = np.clip(
        probability.astype(np.float64),
        0.0,
        1.0,
    )
    target_bool = target.astype(bool)

    target_mean = float(
        target_bool.mean()
    )

    if target_mean == 0.0:
        return float(
            np.clip(
                1.0 - probability.mean(),
                0.0,
                1.0,
            )
        )

    if target_mean == 1.0:
        return float(
            np.clip(
                probability.mean(),
                0.0,
                1.0,
            )
        )

    object_score = _s_object_measure(
        probability,
        target_bool,
    )
    region_score = _s_region_measure(
        probability,
        target_bool,
    )

    score = (
        alpha * object_score
        + (1.0 - alpha) * region_score
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )


def e_measure_single(
    prediction: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """
    Enhanced-alignment E-measure E_xi for one binary foreground map.

    The prediction supplied here is binarized using the frozen deployment
    threshold selected on validation data.
    """
    prediction = prediction.astype(
        np.float64
    )
    target = target.astype(
        np.float64
    )

    target_mean = float(target.mean())

    if target_mean == 0.0:
        return float(
            np.clip(
                1.0 - prediction.mean(),
                0.0,
                1.0,
            )
        )

    if target_mean == 1.0:
        return float(
            np.clip(
                prediction.mean(),
                0.0,
                1.0,
            )
        )

    prediction_centered = (
        prediction - prediction.mean()
    )
    target_centered = (
        target - target.mean()
    )

    alignment = (
        2.0
        * prediction_centered
        * target_centered
        / (
            prediction_centered ** 2
            + target_centered ** 2
            + eps
        )
    )

    enhanced_alignment = (
        (alignment + 1.0) ** 2
        / 4.0
    )

    return float(
        np.clip(
            enhanced_alignment.mean(),
            0.0,
            1.0,
        )
    )


def per_image_foreground_map_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> Dict[str, np.ndarray]:
    """
    Calculate F_beta^omega, E_xi, and S_alpha per image.

    - F_beta^omega and S_alpha use the soft probability map.
    - E_xi uses the binary map produced by the frozen validation threshold.
    - MAE is already computed in per_image_binary_metrics using the soft map.
    """
    probabilities_np = (
        probabilities
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    targets_np = (
        (targets > 0.5)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    predictions_np = (
        probabilities_np >= threshold
    ).astype(np.uint8)

    weighted_f_scores = []
    e_measure_scores = []
    s_measure_scores = []

    for probability, prediction, target in zip(
        probabilities_np,
        predictions_np,
        targets_np,
    ):
        probability_2d = probability[0]
        prediction_2d = prediction[0]
        target_2d = target[0]

        weighted_f_scores.append(
            weighted_f_measure_single(
                probability_2d,
                target_2d,
            )
        )
        e_measure_scores.append(
            e_measure_single(
                prediction_2d,
                target_2d,
            )
        )
        s_measure_scores.append(
            s_measure_single(
                probability_2d,
                target_2d,
            )
        )

    return {
        "weighted_f_measure": np.asarray(
            weighted_f_scores,
            dtype=np.float32,
        ),
        "e_measure": np.asarray(
            e_measure_scores,
            dtype=np.float32,
        ),
        "s_measure": np.asarray(
            s_measure_scores,
            dtype=np.float32,
        ),
    }


def binary_boundary(
    binary_mask: np.ndarray,
) -> np.ndarray:
    binary_mask = (
        binary_mask > 0
    ).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    eroded = cv2.erode(
        binary_mask,
        kernel,
        iterations=1,
    )

    return (
        binary_mask - eroded
    ).clip(0, 1)


def boundary_f1_single(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance: int,
    eps: float = 1e-7,
) -> float:
    pred_boundary = binary_boundary(
        prediction
    )
    target_boundary = binary_boundary(
        target
    )

    kernel_size = (
        2 * tolerance + 1
    )
    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.uint8,
    )

    dilated_target = cv2.dilate(
        target_boundary,
        kernel,
        iterations=1,
    )
    dilated_pred = cv2.dilate(
        pred_boundary,
        kernel,
        iterations=1,
    )

    pred_count = float(
        pred_boundary.sum()
    )
    target_count = float(
        target_boundary.sum()
    )

    if pred_count == 0 and target_count == 0:
        return 1.0
    if pred_count == 0 or target_count == 0:
        return 0.0

    matched_pred = float(
        (
            pred_boundary
            * dilated_target
        ).sum()
    )

    matched_target = float(
        (
            target_boundary
            * dilated_pred
        ).sum()
    )

    precision = matched_pred / (
        pred_count + eps
    )
    recall = matched_target / (
        target_count + eps
    )

    return float(
        2.0 * precision * recall
        / (precision + recall + eps)
    )


def per_image_boundary_f1(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> np.ndarray:
    predictions = (
        probabilities >= threshold
    ).cpu().numpy().astype(np.uint8)

    targets_np = (
        targets > 0.5
    ).cpu().numpy().astype(np.uint8)

    scores = []

    for prediction, target in zip(
        predictions,
        targets_np,
    ):
        scores.append(
            boundary_f1_single(
                prediction[0],
                target[0],
                BOUNDARY_TOLERANCE_PIXELS,
            )
        )

    return np.asarray(
        scores,
        dtype=np.float32,
    )


def sampled_pixel_auc_ap(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float, float, int]:
    scores = (
        probabilities
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.float32)
    )

    labels = (
        targets
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.uint8)
    )

    if (
        MAX_ROC_AP_PIXELS is not None
        and labels.size
        > MAX_ROC_AP_PIXELS
    ):
        rng = np.random.default_rng(
            FIXED_PIXEL_SAMPLE_SEED
        )
        indices = rng.choice(
            labels.size,
            size=MAX_ROC_AP_PIXELS,
            replace=False,
        )

        labels = labels[indices]
        scores = scores[indices]

    auroc = float(
        roc_auc_score(
            labels,
            scores,
        )
    )

    average_precision = float(
        average_precision_score(
            labels,
            scores,
        )
    )

    positive_fraction = float(
        labels.mean()
    )

    return (
        auroc,
        average_precision,
        positive_fraction,
        int(labels.size),
    )


# ============================================================
# TRAINING AND PREDICTION
# ============================================================
def create_grad_scaler(
    amp_enabled: bool,
):
    try:
        return torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(
            enabled=amp_enabled,
        )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.train()

    totals = {
        "loss": 0.0,
        "primary_dice": 0.0,
        "primary_iou": 0.0,
        "wrong_dice": 0.0,
        "wrong_iou": 0.0,
        "primary_compatibility": 0.0,
        "wrong_compatibility": 0.0,
    }

    component_totals: Dict[str, float] = {}
    total_samples = 0

    need_wrong_pass = (
        USE_COUNTERFACTUAL_TRAINING
        or USE_COMPATIBILITY_BRANCH
    )

    for batch in loader:
        images = batch[0].to(
            device,
            non_blocking=True,
        )
        masks = batch[1].to(
            device,
            non_blocking=True,
        )
        primary_embeddings = batch[2].to(
            device,
            non_blocking=True,
        )
        wrong_embeddings = batch[3].to(
            device,
            non_blocking=True,
        )
        primary_target = batch[4].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            primary_output = model(
                images,
                primary_embeddings,
            )

            wrong_output = (
                model(
                    images,
                    wrong_embeddings,
                )
                if need_wrong_pass
                else None
            )

            loss, components = (
                compute_training_loss(
                    primary_output,
                    wrong_output,
                    masks,
                    primary_target,
                )
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            primary_probability = torch.sigmoid(
                primary_output[
                    "final_logits"
                ]
            )

            primary_dice, primary_iou = (
                batch_dice_iou(
                    primary_probability,
                    masks,
                    threshold=0.5,
                )
            )

            if wrong_output is not None:
                wrong_probability = torch.sigmoid(
                    wrong_output[
                        "final_logits"
                    ]
                )
                wrong_dice, wrong_iou = (
                    batch_dice_iou(
                        wrong_probability,
                        masks,
                        threshold=0.5,
                    )
                )
            else:
                wrong_dice = torch.zeros_like(
                    primary_dice
                )
                wrong_iou = torch.zeros_like(
                    primary_iou
                )

        batch_size = images.size(0)
        total_samples += batch_size

        totals["loss"] += (
            loss.item() * batch_size
        )
        totals["primary_dice"] += (
            primary_dice.mean().item()
            * batch_size
        )
        totals["primary_iou"] += (
            primary_iou.mean().item()
            * batch_size
        )
        totals["wrong_dice"] += (
            wrong_dice.mean().item()
            * batch_size
        )
        totals["wrong_iou"] += (
            wrong_iou.mean().item()
            * batch_size
        )
        totals["primary_compatibility"] += (
            primary_output[
                "compatibility_score"
            ].mean().item()
            * batch_size
        )

        if wrong_output is not None:
            totals["wrong_compatibility"] += (
                wrong_output[
                    "compatibility_score"
                ].mean().item()
                * batch_size
            )

        for name, value in components.items():
            component_totals[name] = (
                component_totals.get(
                    name,
                    0.0,
                )
                + value * batch_size
            )

    result = {
        key: value / total_samples
        for key, value in totals.items()
    }

    result.update(
        {
            f"loss_{key}": (
                value / total_samples
            )
            for key, value
            in component_totals.items()
        }
    )

    return result


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_compatibility = 0.0
    total_samples = 0

    for batch in loader:
        images = batch[0].to(
            device,
            non_blocking=True,
        )
        masks = batch[1].to(
            device,
            non_blocking=True,
        )
        embeddings = batch[2].to(
            device,
            non_blocking=True,
        )
        compatibility_target = batch[3].to(
            device,
            non_blocking=True,
        )

        output = model(
            images,
            embeddings,
        )

        loss = segmentation_loss(
            output["final_logits"],
            masks,
        )

        if USE_COMPATIBILITY_BRANCH:
            loss = (
                loss
                + COMPATIBILITY_LOSS_WEIGHT
                * F.binary_cross_entropy_with_logits(
                    output[
                        "compatibility_logit"
                    ],
                    compatibility_target,
                )
            )

        probabilities = torch.sigmoid(
            output["final_logits"]
        )

        dice, iou = batch_dice_iou(
            probabilities,
            masks,
            threshold,
        )

        batch_size = images.size(0)
        total_samples += batch_size

        total_loss += (
            loss.item() * batch_size
        )
        total_dice += (
            dice.mean().item() * batch_size
        )
        total_iou += (
            iou.mean().item() * batch_size
        )
        total_compatibility += (
            output[
                "compatibility_score"
            ].mean().item()
            * batch_size
        )

    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
        "iou": total_iou / total_samples,
        "compatibility": (
            total_compatibility
            / total_samples
        ),
    }


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    images: torch.Tensor,
    embeddings: torch.Tensor,
    use_tta: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    output = model(
        images,
        embeddings,
    )

    probability = torch.sigmoid(
        output["final_logits"]
    )

    compatibility = output[
        "compatibility_score"
    ]

    if not use_tta:
        return probability, compatibility

    flipped_images = torch.flip(
        images,
        dims=[3],
    )

    flipped_output = model(
        flipped_images,
        embeddings,
    )

    flipped_probability = torch.sigmoid(
        flipped_output[
            "final_logits"
        ]
    )

    flipped_probability = torch.flip(
        flipped_probability,
        dims=[3],
    )

    probability = 0.5 * (
        probability + flipped_probability
    )

    compatibility = 0.5 * (
        compatibility
        + flipped_output[
            "compatibility_score"
        ]
    )

    return probability, compatibility


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool,
):
    model.eval()

    probabilities = []
    masks = []
    compatibility_scores = []
    metadata_rows = []

    for batch in loader:
        images = batch[0].to(
            device,
            non_blocking=True,
        )
        batch_masks = batch[1].to(
            device,
            non_blocking=True,
        )
        embeddings = batch[2].to(
            device,
            non_blocking=True,
        )

        batch_probabilities, batch_compatibility = (
            predict_probabilities(
                model,
                images,
                embeddings,
                use_tta,
            )
        )

        probabilities.append(
            batch_probabilities.cpu()
        )
        masks.append(
            batch_masks.cpu()
        )
        compatibility_scores.append(
            batch_compatibility.cpu()
        )

        filenames = batch[4]
        count_categories = batch[5]
        size_categories = batch[6]
        shape_categories = batch[7]
        prompts = batch[8]

        for values in zip(
            filenames,
            count_categories,
            size_categories,
            shape_categories,
            prompts,
        ):
            metadata_rows.append(
                {
                    "filename": values[0],
                    "count_cat": values[1],
                    "size_cat": values[2],
                    "shape_cat": values[3],
                    "prompt": values[4],
                }
            )

    return (
        torch.cat(
            probabilities,
            dim=0,
        ),
        torch.cat(
            masks,
            dim=0,
        ),
        torch.cat(
            compatibility_scores,
            dim=0,
        ),
        metadata_rows,
    )


# ============================================================
# THRESHOLD SELECTION AND FINAL EVALUATION
# ============================================================
def validation_threshold_search(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, pd.DataFrame]:
    rows = []

    for threshold in np.arange(
        0.05,
        0.951,
        0.01,
    ):
        dice, iou = batch_dice_iou(
            probabilities,
            targets,
            float(threshold),
        )

        rows.append(
            {
                "threshold": float(threshold),
                "mean_dice": (
                    dice.mean().item()
                ),
                "mean_iou": (
                    iou.mean().item()
                ),
            }
        )

    table = pd.DataFrame(rows)
    best_row = table.loc[
        table["mean_dice"].idxmax()
    ]

    return (
        float(best_row["threshold"]),
        table,
    )


def evaluate_mode(
    probabilities: torch.Tensor,
    masks: torch.Tensor,
    compatibility_scores: torch.Tensor,
    metadata_rows: List[dict],
    threshold: float,
    prompt_mode: str,
    seed: int,
) -> Tuple[pd.DataFrame, dict]:
    metrics = per_image_binary_metrics(
        probabilities,
        masks,
        threshold,
    )

    foreground_map_metrics = (
        per_image_foreground_map_metrics(
            probabilities,
            masks,
            threshold,
        )
    )
    metrics.update(
        foreground_map_metrics
    )

    boundary_f1 = per_image_boundary_f1(
        probabilities,
        masks,
        threshold,
    )

    auroc, average_precision, positive_fraction, sampled_pixels = (
        sampled_pixel_auc_ap(
            probabilities,
            masks,
        )
    )

    rows = []

    for index, metadata in enumerate(
        metadata_rows
    ):
        row = dict(metadata)

        for metric_name, values in metrics.items():
            row[metric_name] = float(
                values[index]
            )

        row["boundary_f1"] = float(
            boundary_f1[index]
        )
        row["compatibility_score"] = float(
            compatibility_scores[index]
        )
        row["experiment_id"] = EXPERIMENT_ID
        row["experiment_name"] = (
            EXPERIMENT_NAME
        )
        row["seed"] = seed
        row["prompt_mode"] = prompt_mode
        row["threshold"] = threshold
        row["threshold_selected_on"] = (
            "generic_validation"
            if USE_TEXT_CONDITIONING
            else "image_only_validation"
        )

        rows.append(row)

    result_table = pd.DataFrame(rows)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "seed": seed,
        "prompt_mode": prompt_mode,
        "threshold": threshold,
        "threshold_selected_on": (
            "generic_validation"
            if USE_TEXT_CONDITIONING
            else "image_only_validation"
        ),
        "dice_mean": float(
            result_table["dice"].mean()
        ),
        "dice_std": float(
            result_table["dice"].std(ddof=1)
        ),
        "iou_mean": float(
            result_table["iou"].mean()
        ),
        "iou_std": float(
            result_table["iou"].std(ddof=1)
        ),
        "precision_mean": float(
            result_table["precision"].mean()
        ),
        "recall_mean": float(
            result_table["recall"].mean()
        ),
        "specificity_mean": float(
            result_table[
                "specificity"
            ].mean()
        ),
        "mae_mean": float(
            result_table["mae"].mean()
        ),
        "mae_std": float(
            result_table["mae"].std(ddof=1)
        ),
        "weighted_f_measure_mean": float(
            result_table[
                "weighted_f_measure"
            ].mean()
        ),
        "weighted_f_measure_std": float(
            result_table[
                "weighted_f_measure"
            ].std(ddof=1)
        ),
        "e_measure_mean": float(
            result_table[
                "e_measure"
            ].mean()
        ),
        "e_measure_std": float(
            result_table[
                "e_measure"
            ].std(ddof=1)
        ),
        "s_measure_mean": float(
            result_table[
                "s_measure"
            ].mean()
        ),
        "s_measure_std": float(
            result_table[
                "s_measure"
            ].std(ddof=1)
        ),
        "boundary_f1_mean": float(
            result_table[
                "boundary_f1"
            ].mean()
        ),
        "boundary_f1_std": float(
            result_table[
                "boundary_f1"
            ].std(ddof=1)
        ),
        "pixel_auroc": auroc,
        "pixel_average_precision": (
            average_precision
        ),
        "positive_pixel_fraction": (
            positive_fraction
        ),
        "sampled_pixels_for_auc_ap": (
            sampled_pixels
        ),
        "compatibility_score_mean": (
            float(
                result_table[
                    "compatibility_score"
                ].mean()
            )
            if USE_COMPATIBILITY_BRANCH
            else float("nan")
        ),
    }

    return result_table, summary


# ============================================================
# SPLITS AND SAMPLER
# ============================================================
def create_splits(
    dataframe: pd.DataFrame,
    seed: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    # This is used only if the saved baseline split files do not exist.
    # For the ablation, saved split files are strongly preferred.
    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=0.10,
        random_state=42,
        stratify=dataframe["count_cat"],
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=1.0 / 9.0,
        random_state=42,
        stratify=train_val_df["count_cat"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_or_create_splits(
    dataframe: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    paths = {
        "train": os.path.join(
            BASELINE_SPLIT_DIR,
            "train.csv",
        ),
        "val": os.path.join(
            BASELINE_SPLIT_DIR,
            "val.csv",
        ),
        "test": os.path.join(
            BASELINE_SPLIT_DIR,
            "test.csv",
        ),
    }

    if all(
        os.path.exists(path)
        for path in paths.values()
    ):
        print(
            "Reusing fixed baseline splits "
            f"from {BASELINE_SPLIT_DIR}"
        )

        train_df = pd.read_csv(
            paths["train"]
        )
        val_df = pd.read_csv(
            paths["val"]
        )
        test_df = pd.read_csv(
            paths["test"]
        )
    else:
        print(
            "WARNING: fixed baseline splits were "
            "not found. Creating one deterministic "
            "split with random_state=42."
        )

        train_df, val_df, test_df = (
            create_splits(
                dataframe,
                seed=42,
            )
        )

    return (
        base.add_prompt_column(train_df),
        base.add_prompt_column(val_df),
        base.add_prompt_column(test_df),
    )


def build_weighted_sampler(
    train_df: pd.DataFrame,
    seed: int,
) -> WeightedRandomSampler:
    count_frequency = (
        train_df["count_cat"]
        .value_counts()
        .to_dict()
    )

    size_frequency = (
        train_df["size_cat"]
        .value_counts()
        .to_dict()
    )

    total = len(train_df)
    weights = []

    for _, row in train_df.iterrows():
        count_weight = np.sqrt(
            total
            / count_frequency[
                row["count_cat"]
            ]
        )

        size_weight = (
            total
            / size_frequency[
                row["size_cat"]
            ]
        ) ** 0.25

        weight = np.clip(
            count_weight * size_weight,
            0.25,
            8.0,
        )

        weights.append(float(weight))

    weights_array = np.asarray(
        weights,
        dtype=np.float64,
    )

    weights_array /= weights_array.mean()

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights=torch.as_tensor(
            weights_array,
            dtype=torch.double,
        ),
        num_samples=len(weights_array),
        replacement=True,
        generator=generator,
    )


# ============================================================
# GLOBAL RESULT REGISTRATION
# ============================================================
def register_global_results(
    summaries: List[dict],
) -> None:
    os.makedirs(
        MASTER_OUTPUT_DIR,
        exist_ok=True,
    )

    master_path = os.path.join(
        MASTER_OUTPUT_DIR,
        "all_runs.csv",
    )

    new_rows = pd.DataFrame(summaries)

    if os.path.exists(master_path):
        existing = pd.read_csv(master_path)
        combined = pd.concat(
            [existing, new_rows],
            ignore_index=True,
        )

        combined = combined.drop_duplicates(
            subset=[
                "experiment_id",
                "seed",
                "prompt_mode",
                "training_protocol",
            ],
            keep="last",
        )
    else:
        combined = new_rows

    combined = combined.sort_values(
        by=[
            "experiment_id",
            "seed",
            "prompt_mode",
            "training_protocol",
        ]
    )

    temporary_path = (
        master_path + ".tmp"
    )
    combined.to_csv(
        temporary_path,
        index=False,
    )
    os.replace(
        temporary_path,
        master_path,
    )

    # Also create the aggregated mean ± SD table whenever possible.
    numeric_columns = [
        "dice_mean",
        "iou_mean",
        "precision_mean",
        "recall_mean",
        "specificity_mean",
        "mae_mean",
        "weighted_f_measure_mean",
        "e_measure_mean",
        "s_measure_mean",
        "boundary_f1_mean",
        "pixel_auroc",
        "pixel_average_precision",
    ]

    aggregation = {}

    for column in numeric_columns:
        aggregation[f"{column}_across_seeds_mean"] = (
            column,
            "mean",
        )
        aggregation[f"{column}_across_seeds_std"] = (
            column,
            "std",
        )

    aggregated = (
        combined
        .groupby(
            [
                "experiment_id",
                "experiment_name",
                "prompt_mode",
                "training_protocol",
            ],
            dropna=False,
        )
        .agg(**aggregation)
        .reset_index()
    )

    aggregated.to_csv(
        os.path.join(
            MASTER_OUTPUT_DIR,
            "mean_std_across_seeds.csv",
        ),
        index=False,
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    args = parse_args()

    seed = int(args.seed)
    num_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    use_tta = (
        USE_TEST_TIME_AUGMENTATION
        and not args.no_tta
    )

    clinicdb_metadata_csv = str(args.clinicdb_metadata)
    clinicdb_images_dir = str(args.clinicdb_images_dir)
    clinicdb_masks_dir = str(args.clinicdb_masks_dir)
    clinicdb_split_dir = str(args.clinicdb_split_dir)
    clinicdb_val_fraction = float(args.clinicdb_val_fraction)

    if not 0.0 < clinicdb_val_fraction < 1.0:
        raise ValueError(
            "--clinicdb-val-fraction must be between 0 and 1."
        )

    set_seed(
        seed,
        deterministic=args.deterministic,
    )

    output_dir = os.path.join(
        MASTER_OUTPUT_DIR,
        f"{EXPERIMENT_ID}_{EXPERIMENT_NAME}",
        f"seed_{seed}",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    best_model_path = os.path.join(
        output_dir,
        "best_model.pt",
    )

    base.INCLUDE_LOCATION_IN_PROMPT = True
    base.TEXT_BATCH_SIZE = 32
    base.MAX_TEXT_LENGTH = 64

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = (
        USE_AMP
        and device.type == "cuda"
    )

    print("=" * 72)
    print(
        f"{EXPERIMENT_ID}: "
        f"{EXPERIMENT_NAME}"
    )
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Text conditioning: {USE_TEXT_CONDITIONING}")
    print(
        "Compatibility branch: "
        f"{USE_COMPATIBILITY_BRANCH}"
    )
    print(
        "Counterfactual training: "
        f"{USE_COUNTERFACTUAL_TRAINING}"
    )
    print(
        "Boundary refinement: "
        f"{USE_BOUNDARY_REFINEMENT}"
    )

    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"CSV not found: {CSV_PATH}"
        )
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(
            f"Images folder not found: {IMAGES_DIR}"
        )
    if not os.path.isdir(MASKS_DIR):
        raise FileNotFoundError(
            f"Masks folder not found: {MASKS_DIR}"
        )

    dataframe = pd.read_csv(CSV_PATH)

    required_columns = {
        "image_path",
        "mask_path",
        "polyp_count",
        "count_cat",
        "size_cat",
        "shape_cat",
    }

    missing_columns = (
        required_columns.difference(
            dataframe.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = base.add_prompt_column(
        dataframe
    )

    train_df, val_df, test_df = (
        load_or_create_splits(
            dataframe
        )
    )

    # Add ClinicDB at the dataframe level so the original metadata,
    # structured prompts, weighted sampler, and M0-M6 protocol remain
    # identical. The Kvasir test split stays untouched.
    train_df, val_df, test_df = (
        combine_kvasir_and_clinicdb(
            kvasir_train=train_df,
            kvasir_val=val_df,
            kvasir_test=test_df,
            clinic_metadata_csv=clinicdb_metadata_csv,
            clinic_images_dir=clinicdb_images_dir,
            clinic_masks_dir=clinicdb_masks_dir,
            clinic_split_dir=clinicdb_split_dir,
            clinic_validation_fraction=clinicdb_val_fraction,
            base_module=base,
        )
    )

    train_source_counts = (
        train_df["dataset_name"]
        .value_counts()
        .to_dict()
    )
    val_source_counts = (
        val_df["dataset_name"]
        .value_counts()
        .to_dict()
    )
    test_source_counts = (
        test_df["dataset_name"]
        .value_counts()
        .to_dict()
    )

    print("\nTraining sources")
    print(train_df["dataset_name"].value_counts())
    print("\nValidation sources")
    print(val_df["dataset_name"].value_counts())
    print("\nTest sources")
    print(test_df["dataset_name"].value_counts())

    train_df.to_csv(
        os.path.join(
            output_dir,
            "train.csv",
        ),
        index=False,
    )
    val_df.to_csv(
        os.path.join(
            output_dir,
            "val.csv",
        ),
        index=False,
    )
    test_df.to_csv(
        os.path.join(
            output_dir,
            "test.csv",
        ),
        index=False,
    )

    print(
        f"Samples: train={len(train_df)}, "
        f"validation={len(val_df)}, "
        f"test={len(test_df)}"
    )

    if USE_TEXT_CONDITIONING:
        all_prompts = (
            train_df[
                "oracle_prompt"
            ].tolist()
            + val_df[
                "oracle_prompt"
            ].tolist()
            + test_df[
                "oracle_prompt"
            ].tolist()
            + [GENERIC_PROMPT]
        )

        prompt_embeddings, text_dim = (
            base.build_prompt_embedding_cache(
                prompts=all_prompts,
                model_name=TEXT_MODEL_NAME,
                cache_path=TEXT_CACHE_PATH,
                device=device,
            )
        )
    else:
        prompt_embeddings = {}
        text_dim = 1

    train_dataset = AblationTrainingDataset(
        dataframe=train_df,
        prompt_embeddings=prompt_embeddings,
        images_root=IMAGES_DIR,
        masks_root=MASKS_DIR,
        image_size=IMAGE_SIZE,
        augment=True,
        generic_prompt_probability=(
            GENERIC_PROMPT_PROBABILITY
        ),
        seed=seed,
        text_dim=text_dim,
    )

    prompt_modes = (
        ["oracle", "generic", "wrong"]
        if USE_TEXT_CONDITIONING
        else ["image_only"]
    )

    validation_datasets = {
        mode: AblationEvaluationDataset(
            dataframe=val_df,
            prompt_embeddings=prompt_embeddings,
            images_root=IMAGES_DIR,
            masks_root=MASKS_DIR,
            image_size=IMAGE_SIZE,
            prompt_mode=mode,
            seed=seed,
            text_dim=text_dim,
        )
        for mode in prompt_modes
    }

    test_datasets = {
        mode: AblationEvaluationDataset(
            dataframe=test_df,
            prompt_embeddings=prompt_embeddings,
            images_root=IMAGES_DIR,
            masks_root=MASKS_DIR,
            image_size=IMAGE_SIZE,
            prompt_mode=mode,
            seed=seed + 1,
            text_dim=text_dim,
        )
        for mode in prompt_modes
    }

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    loader_common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": (
            device.type == "cuda"
        ),
        "persistent_workers": (
            num_workers > 0
        ),
        "worker_init_fn": seed_worker,
        "generator": loader_generator,
    }

    if USE_WEIGHTED_SAMPLER:
        train_loader = DataLoader(
            train_dataset,
            sampler=build_weighted_sampler(
                train_df,
                seed,
            ),
            shuffle=False,
            drop_last=False,
            **loader_common,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=False,
            **loader_common,
        )

    validation_loaders = {
        mode: DataLoader(
            dataset,
            shuffle=False,
            drop_last=False,
            **loader_common,
        )
        for mode, dataset
        in validation_datasets.items()
    }

    test_loaders = {
        mode: DataLoader(
            dataset,
            shuffle=False,
            drop_last=False,
            **loader_common,
        )
        for mode, dataset
        in test_datasets.items()
    }

    sample_batch = next(
        iter(train_loader)
    )

    print(
        "Image batch:",
        tuple(sample_batch[0].shape),
    )
    print(
        "Mask batch:",
        tuple(sample_batch[1].shape),
    )
    print(
        "Text batch:",
        tuple(sample_batch[2].shape),
    )

    model = AblationUNet(
        text_dim=text_dim,
        in_channels=3,
        out_channels=1,
        base_channels=BASE_CHANNELS,
    ).to(device)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=6,
            threshold=1e-4,
            min_lr=MIN_LEARNING_RATE,
        )
    )

    scaler = create_grad_scaler(
        amp_enabled
    )

    history: List[dict] = []
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    training_start = time.perf_counter()

    for epoch in range(
        1,
        num_epochs + 1,
    ):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
        )

        validation_metrics = {
            mode: validate_one_epoch(
                model,
                loader,
                device,
                threshold=0.5,
            )
            for mode, loader
            in validation_loaders.items()
        }

        if USE_TEXT_CONDITIONING:
            robust_score = (
                0.25
                * validation_metrics[
                    "oracle"
                ]["dice"]
                + 0.55
                * validation_metrics[
                    "generic"
                ]["dice"]
                + 0.20
                * validation_metrics[
                    "wrong"
                ]["dice"]
            )
        else:
            robust_score = (
                validation_metrics[
                    "image_only"
                ]["dice"]
            )

        scheduler.step(robust_score)

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        record = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "robust_val_score": robust_score,
            **{
                f"train_{key}": value
                for key, value
                in train_metrics.items()
            },
        }

        for mode, metrics in (
            validation_metrics.items()
        ):
            for key, value in metrics.items():
                record[
                    f"{mode}_val_{key}"
                ] = value

        history.append(record)

        pd.DataFrame(history).to_csv(
            os.path.join(
                output_dir,
                "history.csv",
            ),
            index=False,
        )

        mode_text = " | ".join(
            f"val {mode}={metrics['dice']:.4f}"
            for mode, metrics
            in validation_metrics.items()
        )

        print(
            f"Epoch {epoch:03d}/"
            f"{num_epochs} | "
            f"lr={current_lr:.2e} | "
            f"train Dice="
            f"{train_metrics['primary_dice']:.4f} | "
            f"{mode_text} | "
            f"score={robust_score:.4f}"
        )

        if robust_score > best_score + 1e-4:
            best_score = robust_score
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                best_model_path,
            )

            print(
                "  Saved new best checkpoint."
            )
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print(
                "Early stopping after "
                f"{EARLY_STOPPING_PATIENCE} "
                "epochs without improvement."
            )
            break

    training_seconds = (
        time.perf_counter()
        - training_start
    )

    model.load_state_dict(
        safe_torch_load(
            best_model_path,
            device,
        )
    )

    # --------------------------------------------------------
    # CORRECT THRESHOLD SELECTION:
    # validation only, then frozen for every test mode.
    # --------------------------------------------------------
    threshold_source_mode = (
        "generic"
        if USE_TEXT_CONDITIONING
        else "image_only"
    )

    (
        val_probabilities,
        val_masks,
        _,
        _,
    ) = collect_predictions(
        model,
        validation_loaders[
            threshold_source_mode
        ],
        device,
        use_tta,
    )

    deployment_threshold, threshold_table = (
        validation_threshold_search(
            val_probabilities,
            val_masks,
        )
    )

    threshold_table.to_csv(
        os.path.join(
            output_dir,
            "validation_threshold_search.csv",
        ),
        index=False,
    )

    print(
        "Frozen deployment threshold "
        f"selected on {threshold_source_mode} "
        "validation data: "
        f"{deployment_threshold:.3f}"
    )

    summaries: List[dict] = []

    evaluation_start = time.perf_counter()

    for mode in prompt_modes:
        (
            probabilities,
            masks,
            compatibility_scores,
            metadata_rows,
        ) = collect_predictions(
            model,
            test_loaders[mode],
            device,
            use_tta,
        )

        result_table, summary = (
            evaluate_mode(
                probabilities,
                masks,
                compatibility_scores,
                metadata_rows,
                deployment_threshold,
                mode,
                seed,
            )
        )

        result_table.to_csv(
            os.path.join(
                output_dir,
                f"test_{mode}_per_image.csv",
            ),
            index=False,
        )

        summary[
            "best_epoch"
        ] = best_epoch
        summary[
            "best_robust_validation_score"
        ] = best_score
        summary[
            "training_seconds"
        ] = training_seconds
        summary[
            "parameter_count"
        ] = parameter_count
        summary[
            "trainable_parameter_count"
        ] = trainable_parameter_count
        summary[
            "tta_used"
        ] = use_tta
        summary["training_protocol"] = (
            "Kvasir-SEG+CVC-ClinicDB_trainval__Kvasir-SEG_test"
        )
        summary["kvasir_train_samples"] = int(
            train_source_counts.get("Kvasir-SEG", 0)
        )
        summary["clinicdb_train_samples"] = int(
            train_source_counts.get("CVC-ClinicDB", 0)
        )
        summary["kvasir_val_samples"] = int(
            val_source_counts.get("Kvasir-SEG", 0)
        )
        summary["clinicdb_val_samples"] = int(
            val_source_counts.get("CVC-ClinicDB", 0)
        )
        summary["kvasir_test_samples"] = int(
            test_source_counts.get("Kvasir-SEG", 0)
        )

        summaries.append(summary)

        print(
            f"Test {mode} | "
            f"Dice={summary['dice_mean']:.4f} "
            f"± {summary['dice_std']:.4f} | "
            f"IoU={summary['iou_mean']:.4f} "
            f"± {summary['iou_std']:.4f} | "
            f"Boundary F1="
            f"{summary['boundary_f1_mean']:.4f} | "
            f"MAE={summary['mae_mean']:.4f} | "
            f"F_beta^omega="
            f"{summary['weighted_f_measure_mean']:.4f} | "
            f"E_xi={summary['e_measure_mean']:.4f} | "
            f"S_alpha={summary['s_measure_mean']:.4f} | "
            f"AUROC={summary['pixel_auroc']:.4f} | "
            f"AP={summary['pixel_average_precision']:.4f}"
        )

    evaluation_seconds = (
        time.perf_counter()
        - evaluation_start
    )

    for summary in summaries:
        summary[
            "evaluation_seconds_all_modes"
        ] = evaluation_seconds

    summary_table = pd.DataFrame(
        summaries
    )

    summary_table.to_csv(
        os.path.join(
            output_dir,
            "run_summary.csv",
        ),
        index=False,
    )

    configuration = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "seed": seed,
        "use_text_conditioning": (
            USE_TEXT_CONDITIONING
        ),
        "use_compatibility_branch": (
            USE_COMPATIBILITY_BRANCH
        ),
        "use_counterfactual_training": (
            USE_COUNTERFACTUAL_TRAINING
        ),
        "use_boundary_refinement": (
            USE_BOUNDARY_REFINEMENT
        ),
        "csv_path": CSV_PATH,
        "images_dir": IMAGES_DIR,
        "masks_dir": MASKS_DIR,
        "training_protocol": (
            "Kvasir-SEG+CVC-ClinicDB_trainval__Kvasir-SEG_test"
        ),
        "clinicdb_metadata_csv": clinicdb_metadata_csv,
        "clinicdb_images_dir": clinicdb_images_dir,
        "clinicdb_masks_dir": clinicdb_masks_dir,
        "clinicdb_split_dir": clinicdb_split_dir,
        "clinicdb_validation_fraction": clinicdb_val_fraction,
        "train_source_counts": train_source_counts,
        "validation_source_counts": val_source_counts,
        "test_source_counts": test_source_counts,
        "baseline_split_dir": (
            BASELINE_SPLIT_DIR
        ),
        "image_size": IMAGE_SIZE,
        "batch_size": batch_size,
        "num_epochs_requested": (
            num_epochs
        ),
        "num_workers": num_workers,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "base_channels": BASE_CHANNELS,
        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "generic_prompt_probability": (
            GENERIC_PROMPT_PROBABILITY
        ),
        "coarse_loss_weight": (
            COARSE_LOSS_WEIGHT
        ),
        "boundary_loss_weight": (
            BOUNDARY_LOSS_WEIGHT
        ),
        "wrong_prompt_seg_loss_weight": (
            WRONG_PROMPT_SEG_LOSS_WEIGHT
        ),
        "compatibility_loss_weight": (
            COMPATIBILITY_LOSS_WEIGHT
        ),
        "counterfactual_consistency_weight": (
            COUNTERFACTUAL_CONSISTENCY_WEIGHT
        ),
        "compatibility_margin_weight": (
            COMPATIBILITY_MARGIN_WEIGHT
        ),
        "compatibility_margin": (
            COMPATIBILITY_MARGIN
        ),
        "deployment_threshold": (
            deployment_threshold
        ),
        "threshold_selected_on": (
            threshold_source_mode
            + "_validation"
        ),
        "tta_used": use_tta,
        "deterministic": (
            args.deterministic
        ),
        "parameter_count": parameter_count,
        "trainable_parameter_count": (
            trainable_parameter_count
        ),
        "best_epoch": best_epoch,
        "best_robust_validation_score": (
            best_score
        ),
    }

    save_json(
        os.path.join(
            output_dir,
            "configuration.json",
        ),
        configuration,
    )

    register_global_results(
        summaries
    )

    print("\nRun complete.")
    print(f"Output folder: {output_dir}")
    print(
        "Master results: "
        f"{os.path.join(MASTER_OUTPUT_DIR, 'all_runs.csv')}"
    )
    print(
        "Across-seed summary: "
        f"{os.path.join(MASTER_OUTPUT_DIR, 'mean_std_across_seeds.csv')}"
    )


if __name__ == "__main__":
    main()
