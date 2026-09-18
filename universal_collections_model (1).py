"""Modelo tabular de cobranza con atencion entre productos.

Granularidad esperada: una fila por cliente (o cliente-fecha_corte si esa llave
compuesta es materializada previamente como un ID unico). No hay secuencias.

Cada fila contiene:

* variables numericas globales del cliente;
* un indicador 0/1 de presencia para cada producto;
* el bucket actual de cada producto presente y NA si no existe;
* un target binario TG.

El pipeline implementa configuracion central, auditoria, split antes del ajuste
de estadisticas, preprocesamiento train-only, Dataset/DataLoader, red con tokens
de producto y mascaras, AMP, early stopping, metricas, explicaciones por
atencion/ablation y exportacion del ecosistema completo.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ProductColumns:
    """Columnas que representan una familia de producto en la misma fila."""

    name: str
    presence_col: str
    bucket_col: str
    numeric_cols: tuple[str, ...] = ()


@dataclass
class CFG:
    """Control central del experimento.

    Los valores por defecto corresponden al ejemplo visual del usuario. Deben
    cambiarse ``INPUT_PATH`` e ``ID_COL`` por los nombres reales.
    """

    INPUT_PATH: str = "base_clientes.parquet"
    OUTPUT_DIR: str = "artifacts_tabular_attention"

    ID_COL: str = "ID_CLIENTE"
    DATE_COL: str | None = None
    TARGET_COL: str = "TG"
    GLOBAL_NUMERIC_COLS: tuple[str, ...] = (
        "Variable 1",
        "Variable 2",
        "Variable 3",
        "Variable 4",
    )
    PRODUCTS: tuple[ProductColumns, ...] = (
        ProductColumns("TDC", "TDC", "Bucket:TDC"),
        ProductColumns("HIPOTECA", "HIPOTECA", "Bucket:HP"),
        ProductColumns("AUTO", "AUTO", "Bucket Auto"),
        ProductColumns("SPL", "SPL", "bucket SPL"),
    )
    # Estados permitidos de bucket. El indice 0 del embedding se reserva para
    # producto ausente; los buckets reales se convierten en 1..K.
    BUCKET_VALUES: tuple[int, ...] = (0, 1, 2, 3, 4)
    STRICT_ABSENT_BUCKET_NA: bool = True
    STRICT_ABSENT_FEATURES_NA: bool = True
    ADD_MISSING_INDICATORS: bool = True
    ADD_PORTFOLIO_AGGREGATES: bool = True

    SPLIT_STRATEGY: str = "stratified"  # "stratified" o "temporal"
    TEST_SIZE: float = 0.15
    VALID_SIZE: float = 0.15
    TRAIN_END: str | None = None
    VALID_END: str | None = None

    SEED: int = 42
    BATCH_SIZE: int = 256
    NUM_WORKERS: int = 0
    EPOCHS: int = 50
    LEARNING_RATE: float = 2e-4
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP_NORM: float = 1.0
    USE_AMP: bool = True

    D_MODEL: int = 64
    BUCKET_EMBED_DIM: int = 16
    PRODUCT_EMBED_DIM: int = 16
    NUM_HEADS: int = 4
    NUM_INTERACTION_LAYERS: int = 2
    HIDDEN_DIM: int = 128
    DROPOUT: float = 0.15

    POS_WEIGHT_MAX: float = 20.0
    MONITOR: str = "average_precision"
    EARLY_STOPPING_PATIENCE: int = 8
    MIN_DELTA: float = 1e-4
    TOP_K_FRACTION: float = 0.10

    def __post_init__(self) -> None:
        if not self.PRODUCTS:
            raise ValueError("Debe configurarse al menos un producto.")
        if len({p.name for p in self.PRODUCTS}) != len(self.PRODUCTS):
            raise ValueError("Los nombres de producto deben ser unicos.")
        if len(set(self.BUCKET_VALUES)) != len(self.BUCKET_VALUES):
            raise ValueError("BUCKET_VALUES contiene duplicados.")
        if tuple(sorted(self.BUCKET_VALUES)) != self.BUCKET_VALUES:
            raise ValueError("BUCKET_VALUES debe estar ordenado de menor a mayor.")
        if self.D_MODEL % self.NUM_HEADS:
            raise ValueError("D_MODEL debe ser divisible entre NUM_HEADS.")
        if not 0 < self.VALID_SIZE < 1 or not 0 < self.TEST_SIZE < 1:
            raise ValueError("VALID_SIZE y TEST_SIZE deben estar entre 0 y 1.")
        if self.VALID_SIZE + self.TEST_SIZE >= 1:
            raise ValueError("VALID_SIZE + TEST_SIZE debe ser menor que 1.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CFG":
        values = dict(payload)
        products = []
        for raw_product in values["PRODUCTS"]:
            product = dict(raw_product)
            product["numeric_cols"] = tuple(product.get("numeric_cols", ()))
            products.append(ProductColumns(**product))
        values["PRODUCTS"] = tuple(products)
        values["GLOBAL_NUMERIC_COLS"] = tuple(values["GLOBAL_NUMERIC_COLS"])
        values["BUCKET_VALUES"] = tuple(values["BUCKET_VALUES"])
        return cls(**values)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Fija semillas de Python, NumPy y PyTorch."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def read_table(
    source: str | Path | None = None,
    *,
    sql_query: str | None = None,
    sql_connection: Any | None = None,
) -> pd.DataFrame:
    """Lee CSV, Parquet o una consulta SQL sin gestionar credenciales."""

    if sql_query is not None:
        if sql_connection is None:
            raise ValueError("sql_connection es obligatorio cuando se usa sql_query.")
        return pd.read_sql_query(sql_query, sql_connection)
    if source is None:
        raise ValueError("Debe proporcionarse source o sql_query.")
    path = Path(source)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Formato no soportado: {suffix}. Usa CSV, Parquet o SQL.")


def required_columns(cfg: CFG, include_target: bool = True) -> set[str]:
    columns = {cfg.ID_COL, *cfg.GLOBAL_NUMERIC_COLS}
    if cfg.DATE_COL:
        columns.add(cfg.DATE_COL)
    if include_target:
        columns.add(cfg.TARGET_COL)
    for product in cfg.PRODUCTS:
        columns.update(
            {product.presence_col, product.bucket_col, *product.numeric_cols}
        )
    return columns


def _parse_presence(series: pd.Series, name: str) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{name}: la mascara de presencia contiene NA/no numericos.")
    invalid = ~numeric.isin([0, 1])
    if invalid.any():
        examples = numeric.loc[invalid].drop_duplicates().head().tolist()
        raise ValueError(f"{name}: presencia debe ser 0/1; valores {examples}.")
    return numeric.to_numpy(dtype=np.int64).astype(bool)


def audit_dataframe(
    frame: pd.DataFrame,
    cfg: CFG,
    *,
    require_target: bool = True,
    require_unique_id: bool = True,
) -> dict[str, Any]:
    """Detiene el pipeline ante errores de granularidad o estructura."""

    missing = sorted(required_columns(cfg, require_target).difference(frame.columns))
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}")
    if frame.empty:
        raise ValueError("La base esta vacia.")
    if frame[cfg.ID_COL].isna().any():
        raise ValueError(f"{cfg.ID_COL} contiene identificadores nulos.")
    duplicated = frame[cfg.ID_COL].duplicated(keep=False)
    if require_unique_id and duplicated.any():
        examples = frame.loc[duplicated, cfg.ID_COL].head().tolist()
        raise ValueError(
            "La granularidad esperada es una fila por cliente; se encontraron "
            f"IDs duplicados. Ejemplos: {examples}"
        )

    if require_target:
        target = pd.to_numeric(frame[cfg.TARGET_COL], errors="coerce")
        if target.isna().any() or not target.isin([0, 1]).all():
            raise ValueError(f"{cfg.TARGET_COL} debe ser binario 0/1 y no nulo.")
        if target.nunique() < 2:
            raise ValueError("El target contiene una sola clase.")

    product_count = np.zeros(len(frame), dtype=np.int64)
    for product in cfg.PRODUCTS:
        present = _parse_presence(frame[product.presence_col], product.name)
        bucket = pd.to_numeric(frame[product.bucket_col], errors="coerce")
        product_count += present.astype(np.int64)
        missing_when_present = present & bucket.isna().to_numpy()
        if missing_when_present.any():
            rows = np.flatnonzero(missing_when_present)[:5].tolist()
            raise ValueError(
                f"{product.name}: existe producto pero falta bucket. Filas {rows}."
            )
        if cfg.STRICT_ABSENT_BUCKET_NA:
            bucket_when_absent = (~present) & bucket.notna().to_numpy()
            if bucket_when_absent.any():
                rows = np.flatnonzero(bucket_when_absent)[:5].tolist()
                raise ValueError(
                    f"{product.name}: producto ausente con bucket informado. "
                    f"Debe ser NA. Filas {rows}."
                )
        observed = bucket.loc[present]
        invalid = ~observed.isin(cfg.BUCKET_VALUES)
        if invalid.any():
            examples = observed.loc[invalid].drop_duplicates().head().tolist()
            raise ValueError(
                f"{product.name}: buckets fuera de BUCKET_VALUES: {examples}."
            )
        if cfg.STRICT_ABSENT_FEATURES_NA and product.numeric_cols:
            feature_frame = frame.loc[:, list(product.numeric_cols)]
            informed_when_absent = (
                (~present)[:, None] & feature_frame.notna().to_numpy()
            )
            if informed_when_absent.any():
                rows = np.unique(np.where(informed_when_absent)[0])[:5].tolist()
                raise ValueError(
                    f"{product.name}: producto ausente con variables especificas "
                    f"informadas. Deben ser NA. Filas {rows}."
                )
    if (product_count == 0).any():
        rows = np.flatnonzero(product_count == 0)[:5].tolist()
        raise ValueError(f"Hay clientes sin ningun producto presente. Filas {rows}.")

    if cfg.DATE_COL:
        parsed_date = pd.to_datetime(frame[cfg.DATE_COL], errors="coerce")
        if parsed_date.isna().any():
            raise ValueError(f"{cfg.DATE_COL} contiene fechas invalidas.")

    report: dict[str, Any] = {
        "rows": int(len(frame)),
        "unique_ids": int(frame[cfg.ID_COL].nunique()),
        "product_count_min": int(product_count.min()),
        "product_count_max": int(product_count.max()),
        "product_count_mean": float(product_count.mean()),
    }
    if require_target:
        report["target_rate"] = float(
            pd.to_numeric(frame[cfg.TARGET_COL]).mean()
        )
    return report


def assert_disjoint_ids(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str,
) -> None:
    train_ids = set(train[id_col])
    valid_ids = set(validation[id_col])
    test_ids = set(test[id_col])
    if train_ids & valid_ids or train_ids & test_ids or valid_ids & test_ids:
        raise AssertionError("Existe fuga de IDs entre train, validation y test.")


def split_dataframe(
    frame: pd.DataFrame, cfg: CFG
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide antes de ajustar cualquier media, desviacion o peso."""

    if cfg.SPLIT_STRATEGY == "stratified":
        train_valid, test = train_test_split(
            frame,
            test_size=cfg.TEST_SIZE,
            random_state=cfg.SEED,
            stratify=frame[cfg.TARGET_COL],
        )
        relative_valid = cfg.VALID_SIZE / (1.0 - cfg.TEST_SIZE)
        train, validation = train_test_split(
            train_valid,
            test_size=relative_valid,
            random_state=cfg.SEED,
            stratify=train_valid[cfg.TARGET_COL],
        )
    elif cfg.SPLIT_STRATEGY == "temporal":
        if not cfg.DATE_COL or not cfg.TRAIN_END or not cfg.VALID_END:
            raise ValueError(
                "Split temporal requiere DATE_COL, TRAIN_END y VALID_END."
            )
        dates = pd.to_datetime(frame[cfg.DATE_COL], errors="raise")
        train_end = pd.Timestamp(cfg.TRAIN_END)
        valid_end = pd.Timestamp(cfg.VALID_END)
        if train_end >= valid_end:
            raise ValueError("TRAIN_END debe ser anterior a VALID_END.")
        train = frame.loc[dates <= train_end]
        validation = frame.loc[(dates > train_end) & (dates <= valid_end)]
        test = frame.loc[dates > valid_end]
    else:
        raise ValueError("SPLIT_STRATEGY debe ser 'stratified' o 'temporal'.")

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Alguna particion quedo vacia.")
    assert_disjoint_ids(train, validation, test, cfg.ID_COL)
    return train, validation, test


@dataclass
class NumericStats:
    median: list[float]
    mean: list[float]
    std: list[float]


class TabularPreprocessor:
    """Ajusta estadisticas numericas exclusivamente con train."""

    AGGREGATE_NAMES = (
        "portfolio_product_count_norm",
        "portfolio_max_bucket_norm",
        "portfolio_mean_bucket_norm",
        "portfolio_delinquent_count_norm",
    )

    def __init__(self, cfg: CFG) -> None:
        self.cfg = cfg
        self.global_stats: NumericStats | None = None
        self.product_stats: dict[str, NumericStats] = {}

    @property
    def fitted(self) -> bool:
        return self.global_stats is not None and set(self.product_stats) == {
            product.name for product in self.cfg.PRODUCTS
        }

    @property
    def global_input_dim(self) -> int:
        base = len(self.cfg.GLOBAL_NUMERIC_COLS)
        if self.cfg.ADD_MISSING_INDICATORS:
            base *= 2
        if self.cfg.ADD_PORTFOLIO_AGGREGATES:
            base += len(self.AGGREGATE_NAMES)
        return base

    @property
    def product_input_dims(self) -> dict[str, int]:
        multiplier = 2 if self.cfg.ADD_MISSING_INDICATORS else 1
        return {
            product.name: len(product.numeric_cols) * multiplier
            for product in self.cfg.PRODUCTS
        }

    def fit(self, train: pd.DataFrame) -> "TabularPreprocessor":
        audit_dataframe(train, self.cfg, require_target=True)
        self.global_stats = self._fit_stats(
            self._numeric_matrix(train, self.cfg.GLOBAL_NUMERIC_COLS)
        )
        for product in self.cfg.PRODUCTS:
            present = _parse_presence(train[product.presence_col], product.name)
            product_values = self._numeric_matrix(
                train.loc[present], product.numeric_cols
            )
            self.product_stats[product.name] = self._fit_stats(product_values)
        return self

    def transform(
        self, frame: pd.DataFrame, *, require_target: bool = False
    ) -> dict[str, np.ndarray]:
        if not self.fitted or self.global_stats is None:
            raise RuntimeError("El preprocesador debe ajustarse con train primero.")
        audit_dataframe(
            frame,
            self.cfg,
            require_target=require_target,
            require_unique_id=True,
        )
        raw_numeric = self._numeric_matrix(frame, self.cfg.GLOBAL_NUMERIC_COLS)
        global_encoded = self._apply_stats(raw_numeric, self.global_stats)
        global_parts = [global_encoded]

        product_mask, product_bucket_idx, raw_bucket = self._products(frame)
        product_x: dict[str, np.ndarray] = {}
        product_feature_mask: dict[str, np.ndarray] = {}
        for p_idx, product in enumerate(self.cfg.PRODUCTS):
            raw_product = self._numeric_matrix(frame, product.numeric_cols)
            missing_product = np.isnan(raw_product)
            encoded_product = self._apply_stats(
                raw_product, self.product_stats[product.name]
            )
            # Ausencia estructural: ni valores ni indicadores de missing participan.
            encoded_product[~product_mask[:, p_idx]] = 0.0
            product_x[product.name] = encoded_product
            feature_observed = ~missing_product
            feature_observed[~product_mask[:, p_idx]] = False
            product_feature_mask[product.name] = feature_observed
        if self.cfg.ADD_PORTFOLIO_AGGREGATES:
            global_parts.append(
                self._portfolio_aggregates(product_mask, raw_bucket)
            )
        global_x = np.concatenate(global_parts, axis=1).astype(np.float32)

        target = np.full(len(frame), -1.0, dtype=np.float32)
        target_mask = np.zeros(len(frame), dtype=bool)
        if self.cfg.TARGET_COL in frame.columns:
            target_series = pd.to_numeric(
                frame[self.cfg.TARGET_COL], errors="coerce"
            )
            target_mask = target_series.isin([0, 1]).to_numpy()
            target[target_mask] = target_series.loc[target_mask].to_numpy(np.float32)
        if require_target and not target_mask.all():
            raise ValueError("Se requieren targets TG validos para todas las filas.")

        return {
            "global_x": global_x,
            "product_x": product_x,
            "product_feature_mask": product_feature_mask,
            "product_mask": product_mask,
            "product_bucket_idx": product_bucket_idx,
            "target": target,
            "target_mask": target_mask,
        }

    def fit_transform(self, train: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.fit(train).transform(train, require_target=True)

    def to_dict(self) -> dict[str, Any]:
        if not self.fitted or self.global_stats is None:
            raise RuntimeError("No se puede exportar un preprocesador no ajustado.")
        return {
            "global_stats": asdict(self.global_stats),
            "product_stats": {
                name: asdict(stats) for name, stats in self.product_stats.items()
            },
        }

    @classmethod
    def from_dict(cls, cfg: CFG, payload: Mapping[str, Any]) -> "TabularPreprocessor":
        instance = cls(cfg)
        instance.global_stats = NumericStats(**payload["global_stats"])
        instance.product_stats = {
            name: NumericStats(**stats)
            for name, stats in payload["product_stats"].items()
        }
        return instance

    def _numeric_matrix(
        self, frame: pd.DataFrame, columns: Sequence[str]
    ) -> np.ndarray:
        if not columns:
            return np.empty((len(frame), 0), dtype=np.float64)
        return (
            frame.loc[:, list(columns)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        )

    def _fit_stats(self, values: np.ndarray) -> NumericStats:
        if values.shape[1] == 0:
            return NumericStats([], [], [])
        median = np.nanmedian(values, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        imputed = np.where(np.isnan(values), median[None, :], values)
        mean = imputed.mean(axis=0)
        std = imputed.std(axis=0)
        std = np.where((std > 1e-8) & np.isfinite(std), std, 1.0)
        return NumericStats(median.tolist(), mean.tolist(), std.tolist())

    def _apply_stats(self, values: np.ndarray, stats: NumericStats) -> np.ndarray:
        if values.shape[1] == 0:
            return np.empty((len(values), 0), dtype=np.float32)
        missing = np.isnan(values)
        median = np.asarray(stats.median, dtype=np.float64)
        mean = np.asarray(stats.mean, dtype=np.float64)
        std = np.asarray(stats.std, dtype=np.float64)
        imputed = np.where(missing, median[None, :], values)
        scaled = ((imputed - mean[None, :]) / std[None, :]).astype(np.float32)
        if self.cfg.ADD_MISSING_INDICATORS:
            scaled = np.concatenate(
                [scaled, missing.astype(np.float32)], axis=1
            )
        return scaled.astype(np.float32)

    def _products(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(frame)
        p_count = len(self.cfg.PRODUCTS)
        product_mask = np.zeros((n, p_count), dtype=bool)
        product_bucket_idx = np.zeros((n, p_count), dtype=np.int64)
        raw_bucket = np.full((n, p_count), np.nan, dtype=np.float64)
        bucket_to_idx = {
            float(value): idx + 1 for idx, value in enumerate(self.cfg.BUCKET_VALUES)
        }
        for p_idx, product in enumerate(self.cfg.PRODUCTS):
            present = _parse_presence(frame[product.presence_col], product.name)
            bucket = pd.to_numeric(frame[product.bucket_col], errors="coerce").to_numpy()
            product_mask[:, p_idx] = present
            raw_bucket[present, p_idx] = bucket[present]
            for row in np.flatnonzero(present):
                value = float(bucket[row])
                if value not in bucket_to_idx:
                    raise ValueError(
                        f"{product.name}: bucket {value} no configurado."
                    )
                product_bucket_idx[row, p_idx] = bucket_to_idx[value]
        return product_mask, product_bucket_idx, raw_bucket

    def _portfolio_aggregates(
        self, product_mask: np.ndarray, raw_bucket: np.ndarray
    ) -> np.ndarray:
        p_count = product_mask.shape[1]
        product_count = product_mask.sum(axis=1)
        minimum = float(min(self.cfg.BUCKET_VALUES))
        maximum = float(max(self.cfg.BUCKET_VALUES))
        scale = max(maximum - minimum, 1.0)
        normalized_bucket = (raw_bucket - minimum) / scale
        max_bucket = np.nanmax(normalized_bucket, axis=1)
        mean_bucket = np.nanmean(normalized_bucket, axis=1)
        delinquent = ((raw_bucket > minimum) & product_mask).sum(axis=1)
        return np.column_stack(
            [
                product_count / p_count,
                max_bucket,
                mean_bucket,
                delinquent / p_count,
            ]
        ).astype(np.float32)


class CustomerProductDataset(Dataset[dict[str, Tensor]]):
    """Dataset vectorizado; ``__getitem__`` sólo entrega una fila ya traducida."""

    def __init__(
        self,
        frame: pd.DataFrame,
        preprocessor: TabularPreprocessor,
        *,
        require_target: bool,
    ) -> None:
        arrays = preprocessor.transform(frame, require_target=require_target)
        self.global_x = torch.from_numpy(arrays["global_x"])
        self.product_x = {
            name: torch.from_numpy(values)
            for name, values in arrays["product_x"].items()
        }
        self.product_feature_mask = {
            name: torch.from_numpy(values)
            for name, values in arrays["product_feature_mask"].items()
        }
        self.product_mask = torch.from_numpy(arrays["product_mask"])
        self.product_bucket_idx = torch.from_numpy(arrays["product_bucket_idx"])
        self.target = torch.from_numpy(arrays["target"])
        self.target_mask = torch.from_numpy(arrays["target_mask"])

    def __len__(self) -> int:
        return len(self.global_x)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {
            "global_x": self.global_x[index],
            "product_x": {
                name: values[index] for name, values in self.product_x.items()
            },
            "product_feature_mask": {
                name: values[index]
                for name, values in self.product_feature_mask.items()
            },
            "product_mask": self.product_mask[index],
            "product_bucket_idx": self.product_bucket_idx[index],
            "target": self.target[index],
            "target_mask": self.target_mask[index],
        }


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class ProductInteractionBlock(nn.Module):
    """Self-attention sobre los productos presentes de una misma fila."""

    def __init__(self, d_model: int, num_heads: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(
        self, tokens: Tensor, product_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        attention_output, attention_weights = self.attention(
            tokens,
            tokens,
            tokens,
            key_padding_mask=~product_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = self.norm1(tokens + self.dropout(attention_output))
        tokens = self.norm2(tokens + self.dropout(self.feed_forward(tokens)))
        tokens = tokens * product_mask.unsqueeze(-1).to(tokens.dtype)
        return tokens, attention_weights


class TabularPortfolioAttention(nn.Module):
    """Clasificador binario de una fila con atencion producto-producto."""

    def __init__(
        self,
        *,
        global_input_dim: int,
        product_names: Sequence[str],
        product_input_dims: Mapping[str, int],
        num_bucket_values: int,
        d_model: int,
        bucket_embed_dim: int,
        product_embed_dim: int,
        num_heads: int,
        num_interaction_layers: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.product_names = tuple(product_names)
        self.num_products = len(self.product_names)
        self.d_model = d_model
        self.global_encoder = MLP(
            global_input_dim, d_model, hidden_dim, dropout
        )
        self.bucket_embedding = nn.Embedding(
            num_bucket_values + 1,
            bucket_embed_dim,
            padding_idx=0,
        )
        self.product_embedding = nn.Embedding(
            self.num_products, product_embed_dim
        )
        self.product_encoders = nn.ModuleDict(
            {
                name: MLP(
                    product_input_dims[name]
                    + bucket_embed_dim
                    + product_embed_dim,
                    d_model,
                    hidden_dim,
                    dropout,
                )
                for name in self.product_names
            }
        )
        self.global_to_token = nn.Linear(d_model, d_model, bias=False)
        self.token_norm = nn.LayerNorm(d_model)
        self.interaction_blocks = nn.ModuleList(
            [
                ProductInteractionBlock(
                    d_model, num_heads, hidden_dim, dropout
                )
                for _ in range(num_interaction_layers)
            ]
        )
        self.product_pool = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.branch_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )
        self.fusion = MLP(2 * d_model, d_model, hidden_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Tensor]:
        global_x = batch["global_x"].float()
        product_x: Mapping[str, Tensor] = batch["product_x"]
        product_mask = batch["product_mask"].bool()
        bucket_idx = batch["product_bucket_idx"].long()
        if not product_mask.any(dim=1).all():
            raise ValueError("Cada cliente debe conservar al menos un producto.")

        batch_size = global_x.shape[0]
        global_state = self.global_encoder(global_x)
        product_tokens: list[Tensor] = []
        for p_idx, name in enumerate(self.product_names):
            product_ids = torch.full(
                (batch_size,), p_idx, device=global_x.device, dtype=torch.long
            )
            token_input = torch.cat(
                [
                    product_x[name].float(),
                    self.bucket_embedding(bucket_idx[:, p_idx]),
                    self.product_embedding(product_ids),
                ],
                dim=-1,
            )
            product_tokens.append(self.product_encoders[name](token_input))
        tokens = torch.stack(product_tokens, dim=1)
        tokens = self.token_norm(
            tokens + self.global_to_token(global_state)[:, None, :]
        )
        tokens = tokens * product_mask.unsqueeze(-1).to(tokens.dtype)

        interaction_weights: list[Tensor] = []
        for block in self.interaction_blocks:
            tokens, weights = block(tokens, product_mask)
            interaction_weights.append(weights)

        product_state, pool_weights = self.product_pool(
            global_state[:, None, :],
            tokens,
            tokens,
            key_padding_mask=~product_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        product_state = product_state[:, 0, :]
        product_attention = pool_weights[:, :, 0, :].mean(dim=1)

        branch_weights = torch.softmax(
            self.branch_gate(torch.cat([global_state, product_state], dim=-1)),
            dim=-1,
        )
        gated_state = (
            branch_weights[:, 0:1] * global_state
            + branch_weights[:, 1:2] * product_state
        )
        fused = self.fusion(
            torch.cat([gated_state, global_state * product_state], dim=-1)
        )
        logits = self.head(fused).squeeze(-1)
        return {
            "logits": logits,
            "probability": torch.sigmoid(logits),
            "product_attention": product_attention,
            "branch_weights": branch_weights,
            "interaction_attention": torch.stack(interaction_weights, dim=1),
            "global_state": global_state,
            "product_state": product_state,
        }


def model_kwargs(cfg: CFG, preprocessor: TabularPreprocessor) -> dict[str, Any]:
    return {
        "global_input_dim": preprocessor.global_input_dim,
        "product_names": [p.name for p in cfg.PRODUCTS],
        "product_input_dims": preprocessor.product_input_dims,
        "num_bucket_values": len(cfg.BUCKET_VALUES),
        "d_model": cfg.D_MODEL,
        "bucket_embed_dim": cfg.BUCKET_EMBED_DIM,
        "product_embed_dim": cfg.PRODUCT_EMBED_DIM,
        "num_heads": cfg.NUM_HEADS,
        "num_interaction_layers": cfg.NUM_INTERACTION_LAYERS,
        "hidden_dim": cfg.HIDDEN_DIM,
        "dropout": cfg.DROPOUT,
    }


def compute_pos_weight(target: Sequence[float], maximum: float = 20.0) -> float:
    target_array = np.asarray(target, dtype=float)
    positives = float((target_array == 1).sum())
    negatives = float((target_array == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Train debe contener ambas clases para calcular pos_weight.")
    return float(np.clip(negatives / positives, 1.0, maximum))


def move_batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Mapping):
            moved[key] = {
                name: tensor.to(device) for name, tensor in value.items()
            }
        else:
            moved[key] = value.to(device)
    return moved


@torch.no_grad()
def predict_loader(
    model: TabularPortfolioAttention,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "probability": [],
        "target": [],
        "target_mask": [],
        "product_attention": [],
        "branch_weights": [],
    }
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        output = model(batch)
        for name in ("probability", "product_attention", "branch_weights"):
            collected[name].append(output[name].detach().cpu().numpy())
        collected["target"].append(batch["target"].detach().cpu().numpy())
        collected["target_mask"].append(
            batch["target_mask"].detach().cpu().numpy()
        )
    return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}


def best_f1_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    thresholds = np.unique(np.concatenate([[0.0], probability, [1.0]]))
    f1_values = [
        f1_score(target, probability >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(f1_values))])


def binary_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
    top_k_fraction: float,
) -> dict[str, float]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if np.unique(target).size < 2:
        raise ValueError("Las metricas requieren ambas clases.")
    prediction = probability >= threshold
    fpr, tpr, _ = roc_curve(target, probability)
    top_n = max(1, int(math.ceil(len(target) * top_k_fraction)))
    top_idx = np.argsort(-probability)[:top_n]
    positive_total = max(int((target == 1).sum()), 1)
    return {
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "ks": float(np.max(tpr - fpr)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(target, probability)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target, prediction)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "top_k_fraction": float(top_k_fraction),
        "precision_at_top_k": float(target[top_idx].mean()),
        "capture_at_top_k": float(target[top_idx].sum() / positive_total),
    }


def _monitor_direction(metric_name: str) -> str:
    return "min" if metric_name in {"loss", "log_loss", "brier"} else "max"


def _is_improvement(
    current: float, best: float, direction: str, min_delta: float
) -> bool:
    if direction == "max":
        return current > best + min_delta
    return current < best - min_delta


def fit_model(
    model: TabularPortfolioAttention,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    cfg: CFG,
    device: torch.device,
    pos_weight: float,
) -> tuple[list[dict[str, float]], int, float]:
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    use_amp = bool(cfg.USE_AMP and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    direction = _monitor_direction(cfg.MONITOR)
    best_metric = -math.inf if direction == "max" else math.inf
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    patience = 0
    history: list[dict[str, float]] = []

    for epoch in range(cfg.EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_rows = 0
        for raw_batch in train_loader:
            batch = move_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(batch)
                loss = criterion(output["logits"], batch["target"].float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(batch["target"].shape[0])
            train_loss_sum += float(loss.detach().cpu()) * batch_size
            train_rows += batch_size

        validation_output = predict_loader(model, validation_loader, device)
        valid_target = validation_output["target"].astype(int)
        valid_probability = validation_output["probability"]
        valid_threshold = best_f1_threshold(valid_target, valid_probability)
        valid_metrics = binary_metrics(
            valid_target,
            valid_probability,
            threshold=valid_threshold,
            top_k_fraction=cfg.TOP_K_FRACTION,
        )
        valid_loss = float(
            log_loss(valid_target, valid_probability, labels=[0, 1])
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss_sum / max(train_rows, 1),
            "validation_loss": valid_loss,
            **{f"validation_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(row)

        if cfg.MONITOR == "loss":
            monitored = valid_loss
        elif cfg.MONITOR in valid_metrics:
            monitored = valid_metrics[cfg.MONITOR]
        else:
            raise KeyError(f"Metrica MONITOR desconocida: {cfg.MONITOR}")
        if _is_improvement(monitored, best_metric, direction, cfg.MIN_DELTA):
            best_metric = monitored
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.EARLY_STOPPING_PATIENCE:
                break

    if best_state is None:
        raise RuntimeError("No se pudo seleccionar una mejor epoca.")
    model.load_state_dict(best_state)
    return history, best_epoch, float(best_metric)


@torch.no_grad()
def score_frame(
    model: TabularPortfolioAttention,
    frame: pd.DataFrame,
    preprocessor: TabularPreprocessor,
    cfg: CFG,
    device: torch.device,
    *,
    batch_size: int | None = None,
) -> dict[str, np.ndarray]:
    dataset = CustomerProductDataset(
        frame, preprocessor, require_target=False
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )
    return predict_loader(model, loader, device)


def explain_frame(
    model: TabularPortfolioAttention,
    frame: pd.DataFrame,
    preprocessor: TabularPreprocessor,
    cfg: CFG,
    device: torch.device,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Devuelve score, atencion, gate y delta de ablation por producto."""

    base = score_frame(model, frame, preprocessor, cfg, device)
    result = pd.DataFrame(
        {
            cfg.ID_COL: frame[cfg.ID_COL].to_numpy(),
            "risk_score": base["probability"],
            "prediction": (base["probability"] >= threshold).astype(int),
            "branch_weight_global": base["branch_weights"][:, 0],
            "branch_weight_products": base["branch_weights"][:, 1],
        }
    )
    if cfg.TARGET_COL in frame.columns:
        result[cfg.TARGET_COL] = frame[cfg.TARGET_COL].to_numpy()

    masks = np.column_stack(
        [
            _parse_presence(frame[p.presence_col], p.name)
            for p in cfg.PRODUCTS
        ]
    )
    product_count = masks.sum(axis=1)
    for p_idx, product in enumerate(cfg.PRODUCTS):
        result[f"attention_{product.name}"] = base["product_attention"][:, p_idx]
        delta = np.full(len(frame), np.nan, dtype=np.float64)
        eligible = masks[:, p_idx] & (product_count > 1)
        if eligible.any():
            modified = frame.loc[eligible].copy()
            modified[product.presence_col] = 0
            modified[product.bucket_col] = np.nan
            for feature_col in product.numeric_cols:
                modified[feature_col] = np.nan
            without = score_frame(
                model, modified, preprocessor, cfg, device
            )["probability"]
            delta[eligible] = base["probability"][eligible] - without
        result[f"delta_risk_without_{product.name}"] = delta
    return result


def save_bundle(
    output_dir: str | Path,
    cfg: CFG,
    preprocessor: TabularPreprocessor,
    model: TabularPortfolioAttention,
    *,
    threshold: float,
    training_history: Sequence[Mapping[str, float]],
    summary: Mapping[str, Any],
) -> None:
    """Guarda pesos, configuracion, escalado, umbral, historia y metricas."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cfg.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (destination / "preprocessor.json").write_text(
        json.dumps(preprocessor.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (destination / "model_config.json").write_text(
        json.dumps(model_kwargs(cfg, preprocessor), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (destination / "decision.json").write_text(
        json.dumps({"threshold": threshold}, indent=2), encoding="utf-8"
    )
    (destination / "training_history.json").write_text(
        json.dumps(list(training_history), indent=2), encoding="utf-8"
    )
    (destination / "summary.json").write_text(
        json.dumps(dict(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    torch.save(model.state_dict(), destination / "model_state.pt")


def load_bundle(
    output_dir: str | Path,
    device: torch.device,
) -> tuple[CFG, TabularPreprocessor, TabularPortfolioAttention, float]:
    source = Path(output_dir)
    cfg = CFG.from_dict(json.loads((source / "cfg.json").read_text("utf-8")))
    preprocessor = TabularPreprocessor.from_dict(
        cfg, json.loads((source / "preprocessor.json").read_text("utf-8"))
    )
    kwargs = json.loads((source / "model_config.json").read_text("utf-8"))
    model = TabularPortfolioAttention(**kwargs)
    state = torch.load(
        source / "model_state.pt", map_location=device, weights_only=True
    )
    model.load_state_dict(state)
    model.to(device).eval()
    threshold = float(
        json.loads((source / "decision.json").read_text("utf-8"))["threshold"]
    )
    return cfg, preprocessor, model, threshold


def run_training_pipeline(frame: pd.DataFrame, cfg: CFG) -> dict[str, Any]:
    """Ejecuta la fabrica completa y exporta resultados reproducibles."""

    seed_everything(cfg.SEED)
    audit = audit_dataframe(frame, cfg, require_target=True)
    train, validation, test = split_dataframe(frame, cfg)

    preprocessor = TabularPreprocessor(cfg).fit(train)
    train_dataset = CustomerProductDataset(
        train, preprocessor, require_target=True
    )
    validation_dataset = CustomerProductDataset(
        validation, preprocessor, require_target=True
    )
    test_dataset = CustomerProductDataset(
        test, preprocessor, require_target=True
    )

    generator = torch.Generator().manual_seed(cfg.SEED)
    loader_kwargs = {
        "num_workers": cfg.NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularPortfolioAttention(**model_kwargs(cfg, preprocessor)).to(device)
    pos_weight = compute_pos_weight(
        train[cfg.TARGET_COL].to_numpy(), cfg.POS_WEIGHT_MAX
    )
    history, best_epoch, best_metric = fit_model(
        model,
        train_loader,
        validation_loader,
        cfg,
        device,
        pos_weight,
    )

    validation_output = predict_loader(model, validation_loader, device)
    threshold = best_f1_threshold(
        validation_output["target"].astype(int),
        validation_output["probability"],
    )
    validation_metrics = binary_metrics(
        validation_output["target"].astype(int),
        validation_output["probability"],
        threshold=threshold,
        top_k_fraction=cfg.TOP_K_FRACTION,
    )
    test_output = predict_loader(model, test_loader, device)
    test_metrics = binary_metrics(
        test_output["target"].astype(int),
        test_output["probability"],
        threshold=threshold,
        top_k_fraction=cfg.TOP_K_FRACTION,
    )
    summary = {
        "audit": audit,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "device": str(device),
        "pos_weight": pos_weight,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    save_bundle(
        cfg.OUTPUT_DIR,
        cfg,
        preprocessor,
        model,
        threshold=threshold,
        training_history=history,
        summary=summary,
    )
    explanations = explain_frame(
        model,
        test,
        preprocessor,
        cfg,
        device,
        threshold=threshold,
    )
    explanations.to_csv(
        Path(cfg.OUTPUT_DIR) / "test_predictions_and_explanations.csv",
        index=False,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON opcional con los campos de CFG.",
    )
    arguments = parser.parse_args()
    cfg = CFG()
    if arguments.config:
        cfg = CFG.from_dict(json.loads(arguments.config.read_text("utf-8")))
    frame = read_table(cfg.INPUT_PATH)
    summary = run_training_pipeline(frame, cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
