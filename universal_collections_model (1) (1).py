"""
MODELO UNIVERSAL DE COBRANZA A NIVEL CLIENTE

Granularidad:
    una fila por cliente x fecha de corte

Entrada:
    variables globales del cliente
    variables especificas por producto
    presencia de cada producto
    bucket actual de cada producto

Target:
    TG = 1 si el cliente empeora en los proximos 3 meses
    TG = 0 si no empeora

No hay secuencias en este modelo. Cada fecha de corte es una fotografia tabular.
"""


# ============================================================
# 0) IMPORTS
# ============================================================

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


pd.set_option("display.max_columns", None)


# ============================================================
# 1) RUTAS
# ============================================================

INPUT_PATH = "base_clientes.parquet"
MODEL_PATH = "modelo_universal_cobranza.pth"
SCORES_PATH = "scores_test.csv"


# ============================================================
# 2) CONFIG
# ============================================================


@dataclass
class Config:

    seed: int = 42

    batch_size: int = 256
    num_workers: int = 0
    num_epochs: int = 40

    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    d_model: int = 64
    bucket_embedding_dim: int = 12
    product_embedding_dim: int = 12
    num_heads: int = 4
    attention_layers: int = 2
    dropout: float = 0.15

    early_stopping_patience: int = 7
    use_pos_weight: bool = True

    split_strategy: str = "temporal"
    test_size: float = 0.15
    validation_size: float = 0.15


CFG = Config()


# ============================================================
# 3) COLUMNAS DE IDENTIFICACION Y TARGET
# ============================================================

ID_COL = "ID_CLIENTE"

# Si cada cliente aparece una sola vez, usa:
# DATE_COL = None
DATE_COL = "business_date"

TARGET_COL = "TG"


# ============================================================
# 4) VARIABLES GLOBALES
#
# Estas variables tienen sentido para cualquier cliente,
# independientemente de los productos que posea.
#
# Ejemplos:
#     ingreso
#     saldo de depositos
#     score de buro
#     endeudamiento total
#     antiguedad como cliente
#
# ESTA ES LA LISTA QUE DEBES MODIFICAR.
# ============================================================

GLOBAL_FEATURES = [
    "Variable 1",
    "Variable 2",
    "Variable 3",
    "Variable 4",
]


# ============================================================
# 5) VARIABLES ESPECIFICAS POR PRODUCTO
#
# presence_col:
#     1 = el cliente tiene el producto
#     0 = el cliente no tiene el producto
#
# bucket_col:
#     estado actual del producto en la fecha de corte
#
# feature_cols:
#     variables que solamente tienen sentido si existe
#     ese producto.
#
# EJEMPLO:
#
# "TDC": {
#     "presence_col": "TDC",
#     "bucket_col": "Bucket:TDC",
#     "feature_cols": [
#         "utilizacion_tdc",
#         "saldo_tdc",
#         "pago_minimo_tdc"
#     ]
# }
#
# Si el cliente NO tiene TDC:
#     presence_col = 0
#     bucket embedding = AUSENTE
#     variables TDC = 0 despues del preprocesamiento
#     attention TDC = 0
# ============================================================

PRODUCTS = {

    "TDC": {
        "presence_col": "TDC",
        "bucket_col": "Bucket:TDC",
        "feature_cols": [],
    },

    "HIPOTECA": {
        "presence_col": "HIPOTECA",
        "bucket_col": "Bucket:HP",
        "feature_cols": [],
    },

    "AUTO": {
        "presence_col": "AUTO",
        "bucket_col": "Bucket Auto",
        "feature_cols": [],
    },

    "SPL": {
        "presence_col": "SPL",
        "bucket_col": "bucket SPL",
        "feature_cols": [],
    },
}


PRODUCT_NAMES = list(PRODUCTS.keys())


# ============================================================
# 6) BUCKETS PERMITIDOS
#
# El indice 0 del embedding se reserva para PRODUCTO AUSENTE.
#
# Producto ausente -> indice 0
# Bucket real 0    -> indice 1
# Bucket real 1    -> indice 2
# Bucket real 2    -> indice 3
# etc.
#
# De esta manera el modelo NO confunde:
#
#     no tener TDC
#
# con:
#
#     tener TDC en bucket 0
# ============================================================

BUCKET_VALUES = [0, 1, 2, 3, 4]

BUCKET_TO_INDEX = {
    bucket_value: index + 1
    for index, bucket_value in enumerate(BUCKET_VALUES)
}


# ============================================================
# 7) FECHAS DEL SPLIT
#
# Solo se utilizan cuando:
#
# CFG.split_strategy = "temporal"
# ============================================================

TRAIN_END = pd.Timestamp("2025-12-01")
VALID_END = pd.Timestamp("2026-03-01")
TEST_END = pd.Timestamp("2026-05-01")


# ============================================================
# 8) REPRODUCIBILIDAD
# ============================================================


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


seed_everything(CFG.seed)


# ============================================================
# 9) AUDITORIA DE LA BASE
# ============================================================


def audit_base(
    df,
    require_target=True,
):

    required_columns = [
        ID_COL,
        *GLOBAL_FEATURES,
    ]

    if DATE_COL is not None:
        required_columns.append(DATE_COL)

    if require_target:
        required_columns.append(TARGET_COL)

    for product_name in PRODUCT_NAMES:

        product_config = PRODUCTS[product_name]

        required_columns.append(
            product_config["presence_col"]
        )

        required_columns.append(
            product_config["bucket_col"]
        )

        required_columns.extend(
            product_config["feature_cols"]
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas requeridas: {missing_columns}"
        )

    if len(df) == 0:
        raise ValueError("La base esta vacia.")

    if df[ID_COL].isna().any():
        raise ValueError(
            f"{ID_COL} contiene identificadores nulos."
        )

    # --------------------------------------------------------
    # La llave esperada es:
    #
    # cliente x fecha de corte
    #
    # Si DATE_COL = None, la llave es solamente cliente.
    # --------------------------------------------------------

    key_columns = [ID_COL]

    if DATE_COL is not None:
        key_columns.append(DATE_COL)

    duplicated = df.duplicated(
        subset=key_columns,
        keep=False,
    )

    if duplicated.any():

        examples = (
            df.loc[duplicated, key_columns]
            .head(20)
        )

        raise ValueError(
            "Hay mas de una fila para la misma llave "
            f"{key_columns}.\n\n{examples}"
        )

    if DATE_COL is not None:

        parsed_date = pd.to_datetime(
            df[DATE_COL],
            errors="coerce",
        )

        if parsed_date.isna().any():
            raise ValueError(
                f"{DATE_COL} contiene fechas invalidas."
            )

    if require_target:

        target = pd.to_numeric(
            df[TARGET_COL],
            errors="coerce",
        )

        if target.isna().any():
            raise ValueError(
                f"{TARGET_COL} contiene valores nulos o no numericos."
            )

        if not target.isin([0, 1]).all():
            raise ValueError(
                f"{TARGET_COL} debe contener solamente 0 y 1."
            )

    product_count = np.zeros(
        len(df),
        dtype=np.int64,
    )

    for product_name in PRODUCT_NAMES:

        product_config = PRODUCTS[product_name]

        presence = pd.to_numeric(
            df[product_config["presence_col"]],
            errors="coerce",
        )

        if presence.isna().any():
            raise ValueError(
                f"{product_name}: presencia contiene NA."
            )

        if not presence.isin([0, 1]).all():
            raise ValueError(
                f"{product_name}: presencia debe ser 0 o 1."
            )

        present_mask = presence.to_numpy(dtype=bool)

        product_count += present_mask.astype(np.int64)

        bucket = pd.to_numeric(
            df[product_config["bucket_col"]],
            errors="coerce",
        )

        if bucket.loc[present_mask].isna().any():
            raise ValueError(
                f"{product_name}: falta bucket para un producto existente."
            )

        invalid_bucket = (
            ~bucket.loc[present_mask]
            .isin(BUCKET_VALUES)
        )

        if invalid_bucket.any():

            invalid_values = (
                bucket.loc[present_mask]
                .loc[invalid_bucket]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                f"{product_name}: buckets no permitidos "
                f"{invalid_values}."
            )

    if (product_count == 0).any():
        raise ValueError(
            "Hay clientes sin ningun producto presente."
        )

    report = {
        "rows": int(len(df)),
        "unique_clients": int(df[ID_COL].nunique()),
        "minimum_products": int(product_count.min()),
        "maximum_products": int(product_count.max()),
        "mean_products": float(product_count.mean()),
    }

    if require_target:
        report["target_rate"] = float(
            pd.to_numeric(df[TARGET_COL]).mean()
        )

    return report


# ============================================================
# 10) SPLIT
#
# IMPORTANTE:
#
# El split ocurre ANTES de calcular medianas, medias,
# desviaciones o cualquier parametro de preprocesamiento.
# ============================================================


def split_base(df):

    if CFG.split_strategy == "temporal":

        if DATE_COL is None:
            raise ValueError(
                "El split temporal requiere DATE_COL."
            )

        dates = pd.to_datetime(
            df[DATE_COL],
            errors="raise",
        )

        train = df.loc[
            dates < TRAIN_END
        ].copy()

        validation = df.loc[
            (dates >= TRAIN_END)
            &
            (dates < VALID_END)
        ].copy()

        test = df.loc[
            (dates >= VALID_END)
            &
            (dates < TEST_END)
        ].copy()

    elif CFG.split_strategy == "stratified":

        train_validation, test = train_test_split(
            df,
            test_size=CFG.test_size,
            random_state=CFG.seed,
            stratify=df[TARGET_COL],
        )

        relative_validation_size = (
            CFG.validation_size
            /
            (1.0 - CFG.test_size)
        )

        train, validation = train_test_split(
            train_validation,
            test_size=relative_validation_size,
            random_state=CFG.seed,
            stratify=train_validation[TARGET_COL],
        )

    else:
        raise ValueError(
            "split_strategy debe ser temporal o stratified."
        )

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "TRAIN, VALID o TEST quedo vacio."
        )

    return train, validation, test


# ============================================================
# 11) APRENDER PREPROCESAMIENTO SOLO CON TRAIN
#
# VARIABLES GLOBALES:
#     se ajustan con todas las filas de TRAIN.
#
# VARIABLES DE TDC:
#     se ajustan solamente con clientes que tienen TDC.
#
# VARIABLES DE AUTO:
#     se ajustan solamente con clientes que tienen AUTO.
#
# Y asi sucesivamente.
# ============================================================


def fit_preprocessing(train_df):

    preprocessing = {
        "global": {},
        "products": {},
    }

    # --------------------------------------------------------
    # Variables globales
    # --------------------------------------------------------

    global_values = (
        train_df[GLOBAL_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )

    global_median = np.nanmedian(
        global_values,
        axis=0,
    )

    global_median = np.where(
        np.isfinite(global_median),
        global_median,
        0.0,
    )

    global_imputed = np.where(
        np.isnan(global_values),
        global_median,
        global_values,
    )

    global_mean = global_imputed.mean(axis=0)
    global_std = global_imputed.std(axis=0)

    global_std = np.where(
        (global_std > 1e-8) & np.isfinite(global_std),
        global_std,
        1.0,
    )

    preprocessing["global"] = {
        "median": global_median.tolist(),
        "mean": global_mean.tolist(),
        "std": global_std.tolist(),
    }

    # --------------------------------------------------------
    # Variables especificas por producto
    # --------------------------------------------------------

    for product_name in PRODUCT_NAMES:

        product_config = PRODUCTS[product_name]
        feature_columns = product_config["feature_cols"]

        if len(feature_columns) == 0:

            preprocessing["products"][product_name] = {
                "median": [],
                "mean": [],
                "std": [],
            }

            continue

        present_mask = (
            pd.to_numeric(
                train_df[product_config["presence_col"]],
                errors="raise",
            )
            .to_numpy(dtype=bool)
        )

        if present_mask.sum() == 0:
            raise ValueError(
                f"{product_name}: no hay ejemplos presentes en TRAIN."
            )

        product_values = (
            train_df.loc[present_mask, feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        )

        product_median = np.nanmedian(
            product_values,
            axis=0,
        )

        product_median = np.where(
            np.isfinite(product_median),
            product_median,
            0.0,
        )

        product_imputed = np.where(
            np.isnan(product_values),
            product_median,
            product_values,
        )

        product_mean = product_imputed.mean(axis=0)
        product_std = product_imputed.std(axis=0)

        product_std = np.where(
            (product_std > 1e-8) & np.isfinite(product_std),
            product_std,
            1.0,
        )

        preprocessing["products"][product_name] = {
            "median": product_median.tolist(),
            "mean": product_mean.tolist(),
            "std": product_std.tolist(),
        }

    return preprocessing


# ============================================================
# 12) TRANSFORMAR UNA BASE
#
# Salidas:
#
# X_global:
#     variables globales estandarizadas
#     indicadores de missing
#     resumen del portafolio
#
# X_products:
#     variables especificas de cada producto
#
# product_mask:
#     indica que productos existen
#
# bucket_index:
#     bucket actual de cada producto
#
# y:
#     target binario
# ============================================================


def transform_base(
    df,
    preprocessing,
    require_target=True,
):

    n_rows = len(df)
    n_products = len(PRODUCT_NAMES)

    # --------------------------------------------------------
    # Variables globales
    # --------------------------------------------------------

    global_values = (
        df[GLOBAL_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )

    global_missing = np.isnan(
        global_values
    ).astype(np.float32)

    global_median = np.asarray(
        preprocessing["global"]["median"],
        dtype=np.float64,
    )

    global_mean = np.asarray(
        preprocessing["global"]["mean"],
        dtype=np.float64,
    )

    global_std = np.asarray(
        preprocessing["global"]["std"],
        dtype=np.float64,
    )

    global_imputed = np.where(
        np.isnan(global_values),
        global_median,
        global_values,
    )

    global_scaled = (
        (global_imputed - global_mean)
        /
        global_std
    )

    # --------------------------------------------------------
    # Presencia, buckets y variables por producto
    # --------------------------------------------------------

    product_mask = np.zeros(
        (n_rows, n_products),
        dtype=bool,
    )

    bucket_index = np.zeros(
        (n_rows, n_products),
        dtype=np.int64,
    )

    raw_bucket = np.zeros(
        (n_rows, n_products),
        dtype=np.float32,
    )

    X_products = {}

    for product_position, product_name in enumerate(PRODUCT_NAMES):

        product_config = PRODUCTS[product_name]
        feature_columns = product_config["feature_cols"]

        presence = (
            pd.to_numeric(
                df[product_config["presence_col"]],
                errors="raise",
            )
            .to_numpy(dtype=bool)
        )

        product_mask[:, product_position] = presence

        bucket = pd.to_numeric(
            df[product_config["bucket_col"]],
            errors="coerce",
        )

        if bucket.loc[presence].isna().any():
            raise ValueError(
                f"{product_name}: falta bucket para producto presente."
            )

        for bucket_value, embedding_index in BUCKET_TO_INDEX.items():

            rows = (
                presence
                &
                bucket.eq(bucket_value).to_numpy()
            )

            bucket_index[
                rows,
                product_position,
            ] = embedding_index

            raw_bucket[
                rows,
                product_position,
            ] = float(bucket_value)

        # ----------------------------------------------------
        # Sin variables numericas especificas.
        #
        # El producto sigue teniendo:
        #     embedding de identidad
        #     embedding de bucket
        # ----------------------------------------------------

        if len(feature_columns) == 0:

            X_products[product_name] = np.empty(
                (n_rows, 0),
                dtype=np.float32,
            )

            continue

        product_values = (
            df[feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float64)
        )

        product_missing = np.isnan(
            product_values
        ).astype(np.float32)

        product_stats = (
            preprocessing["products"][product_name]
        )

        product_median = np.asarray(
            product_stats["median"],
            dtype=np.float64,
        )

        product_mean = np.asarray(
            product_stats["mean"],
            dtype=np.float64,
        )

        product_std = np.asarray(
            product_stats["std"],
            dtype=np.float64,
        )

        product_imputed = np.where(
            np.isnan(product_values),
            product_median,
            product_values,
        )

        product_scaled = (
            (product_imputed - product_mean)
            /
            product_std
        )

        product_encoded = np.concatenate(
            [
                product_scaled,
                product_missing,
            ],
            axis=1,
        ).astype(np.float32)

        # ====================================================
        # REGLA CENTRAL DE LA MASCARA
        #
        # Si el cliente NO tiene el producto:
        #
        # todas las variables de esa rama se hacen cero.
        #
        # Por tanto, una variable como utilizacion_tdc
        # NO se evalua cuando TDC = 0.
        # ====================================================

        product_encoded[~presence] = 0.0

        X_products[product_name] = product_encoded

    if not product_mask.any(axis=1).all():
        raise ValueError(
            "Cada cliente debe tener al menos un producto presente."
        )

    # --------------------------------------------------------
    # Resumen del portafolio
    # --------------------------------------------------------

    product_count = product_mask.sum(
        axis=1
    ).astype(np.float32)

    safe_bucket = np.where(
        product_mask,
        raw_bucket,
        0.0,
    )

    maximum_bucket = safe_bucket.max(axis=1)

    mean_bucket = (
        safe_bucket.sum(axis=1)
        /
        product_count
    )

    delinquent_count = (
        (safe_bucket > 0)
        &
        product_mask
    ).sum(axis=1).astype(np.float32)

    maximum_allowed_bucket = max(
        max(BUCKET_VALUES),
        1,
    )

    portfolio_features = np.column_stack(
        [
            product_count / n_products,
            maximum_bucket / maximum_allowed_bucket,
            mean_bucket / maximum_allowed_bucket,
            delinquent_count / product_count,
        ]
    ).astype(np.float32)

    X_global = np.concatenate(
        [
            global_scaled,
            global_missing,
            portfolio_features,
        ],
        axis=1,
    ).astype(np.float32)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = np.full(
        n_rows,
        -1.0,
        dtype=np.float32,
    )

    if TARGET_COL in df.columns:

        target = pd.to_numeric(
            df[TARGET_COL],
            errors="coerce",
        )

        valid_target = target.isin([0, 1])

        y[valid_target] = (
            target.loc[valid_target]
            .to_numpy(dtype=np.float32)
        )

    if require_target and (y < 0).any():
        raise ValueError(
            f"{TARGET_COL} debe ser 0/1 para todas las filas."
        )

    return (
        X_global,
        X_products,
        product_mask,
        bucket_index,
        y,
    )


# ============================================================
# 13) DATASET
# ============================================================


class CustomerPortfolioDataset(Dataset):

    def __init__(
        self,
        X_global,
        X_products,
        product_mask,
        bucket_index,
        y,
    ):

        self.X_global = torch.tensor(
            X_global,
            dtype=torch.float32,
        )

        self.X_products = {
            product_name: torch.tensor(
                product_values,
                dtype=torch.float32,
            )
            for product_name, product_values in X_products.items()
        }

        self.product_mask = torch.tensor(
            product_mask,
            dtype=torch.bool,
        )

        self.bucket_index = torch.tensor(
            bucket_index,
            dtype=torch.long,
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32,
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        return {
            "X_global": self.X_global[index],
            "X_products": {
                product_name: values[index]
                for product_name, values in self.X_products.items()
            },
            "product_mask": self.product_mask[index],
            "bucket_index": self.bucket_index[index],
            "y": self.y[index],
        }


# ============================================================
# 14) MODELO
#
# FLUJO:
#
# variables globales
#         -> encoder global
#
# variables TDC + bucket TDC
#         -> encoder TDC
#
# variables AUTO + bucket AUTO
#         -> encoder AUTO
#
# etc.
#
# tokens de productos presentes
#         -> atencion entre productos
#         -> pooling condicionado por el cliente
#
# estado global + estado productos
#         -> gate aprendido
#         -> TG
# ============================================================


class UniversalCollectionsModel(nn.Module):

    def __init__(
        self,
        global_input_dim,
        product_input_dims,
        product_names,
        number_of_bucket_values,
        d_model=64,
        bucket_embedding_dim=12,
        product_embedding_dim=12,
        num_heads=4,
        attention_layers=2,
        dropout=0.15,
    ):

        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model debe ser divisible entre num_heads."
            )

        self.product_names = list(product_names)

        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.bucket_embedding = nn.Embedding(
            number_of_bucket_values + 1,
            bucket_embedding_dim,
            padding_idx=0,
        )

        self.product_embedding = nn.Embedding(
            len(self.product_names),
            product_embedding_dim,
        )

        self.product_encoders = nn.ModuleDict()

        for product_name in self.product_names:

            product_numeric_dim = (
                product_input_dims[product_name]
            )

            product_encoder_input_dim = (
                product_numeric_dim
                +
                bucket_embedding_dim
                +
                product_embedding_dim
            )

            self.product_encoders[product_name] = nn.Sequential(
                nn.Linear(product_encoder_input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.global_to_products = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.interaction_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(attention_layers)
            ]
        )

        self.interaction_norms = nn.ModuleList(
            [
                nn.LayerNorm(d_model)
                for _ in range(attention_layers)
            ]
        )

        self.product_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.global_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch):

        X_global = batch["X_global"].float()
        X_products = batch["X_products"]
        product_mask = batch["product_mask"].bool()
        bucket_index = batch["bucket_index"].long()

        if not product_mask.any(dim=1).all():
            raise ValueError(
                "Cada cliente debe tener al menos un producto."
            )

        batch_size = X_global.shape[0]

        global_state = self.global_encoder(
            X_global
        )

        product_tokens = []

        for product_position, product_name in enumerate(self.product_names):

            product_ids = torch.full(
                (batch_size,),
                product_position,
                dtype=torch.long,
                device=X_global.device,
            )

            product_input = torch.cat(
                [
                    X_products[product_name].float(),
                    self.bucket_embedding(
                        bucket_index[:, product_position]
                    ),
                    self.product_embedding(product_ids),
                ],
                dim=1,
            )

            product_token = self.product_encoders[
                product_name
            ](
                product_input
            )

            product_tokens.append(
                product_token
            )

        product_tokens = torch.stack(
            product_tokens,
            dim=1,
        )

        # El estado general del cliente contextualiza cada producto.
        product_tokens = (
            product_tokens
            +
            self.global_to_products(global_state)[:, None, :]
        )

        # Producto ausente = token cero.
        product_tokens = (
            product_tokens
            *
            product_mask.unsqueeze(-1)
        )

        # ----------------------------------------------------
        # Interaccion producto-producto
        # ----------------------------------------------------

        for attention_layer, layer_norm in zip(
            self.interaction_layers,
            self.interaction_norms,
        ):

            attention_update, _ = attention_layer(
                query=product_tokens,
                key=product_tokens,
                value=product_tokens,
                key_padding_mask=~product_mask,
                need_weights=False,
            )

            product_tokens = layer_norm(
                product_tokens
                +
                attention_update
            )

            product_tokens = (
                product_tokens
                *
                product_mask.unsqueeze(-1)
            )

        # ----------------------------------------------------
        # El estado global pregunta:
        #
        # que productos son mas importantes para este cliente
        # en esta fecha de corte.
        # ----------------------------------------------------

        pooled_products, product_attention = self.product_pooling(
            query=global_state[:, None, :],
            key=product_tokens,
            value=product_tokens,
            key_padding_mask=~product_mask,
            need_weights=True,
        )

        product_state = pooled_products[:, 0, :]
        product_attention = product_attention[:, 0, :]

        # ----------------------------------------------------
        # Gate dinamico
        #
        # global_weight cercano a 1:
        #     pesan mas variables globales
        #
        # global_weight cercano a 0:
        #     pesan mas los productos
        # ----------------------------------------------------

        global_weight = self.global_gate(
            torch.cat(
                [
                    global_state,
                    product_state,
                ],
                dim=1,
            )
        )

        final_state = (
            global_weight * global_state
            +
            (1.0 - global_weight) * product_state
        )

        logits = self.output_head(
            final_state
        ).squeeze(1)

        return {
            "logits": logits,
            "probability": torch.sigmoid(logits),
            "product_attention": product_attention,
            "global_weight": global_weight.squeeze(1),
        }


# ============================================================
# 15) METRICAS
# ============================================================


def ks_statistic(
    y_true,
    y_score,
):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_score,
    )

    return float(
        np.max(tpr - fpr)
    )


@torch.no_grad()
def predict_loader(
    model,
    loader,
    device,
):

    model.eval()

    all_logits = []
    all_probabilities = []
    all_targets = []
    all_attention = []
    all_global_weight = []

    for batch in loader:

        batch = {
            key: (
                {
                    product_name: tensor.to(device)
                    for product_name, tensor in value.items()
                }
                if isinstance(value, dict)
                else value.to(device)
            )
            for key, value in batch.items()
        }

        output = model(batch)

        all_logits.append(
            output["logits"]
            .detach()
            .cpu()
            .numpy()
        )

        all_probabilities.append(
            output["probability"]
            .detach()
            .cpu()
            .numpy()
        )

        all_targets.append(
            batch["y"]
            .detach()
            .cpu()
            .numpy()
        )

        all_attention.append(
            output["product_attention"]
            .detach()
            .cpu()
            .numpy()
        )

        all_global_weight.append(
            output["global_weight"]
            .detach()
            .cpu()
            .numpy()
        )

    return (
        np.concatenate(all_logits),
        np.concatenate(all_probabilities),
        np.concatenate(all_targets),
        np.concatenate(all_attention),
        np.concatenate(all_global_weight),
    )


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
    threshold=0.50,
):

    model.eval()

    total_loss = 0.0
    n_observations = 0

    for batch in loader:

        batch = {
            key: (
                {
                    product_name: tensor.to(device)
                    for product_name, tensor in value.items()
                }
                if isinstance(value, dict)
                else value.to(device)
            )
            for key, value in batch.items()
        }

        output = model(batch)

        loss = criterion(
            output["logits"],
            batch["y"],
        )

        batch_size = batch["y"].shape[0]

        total_loss += loss.item() * batch_size
        n_observations += batch_size

    average_loss = (
        total_loss
        /
        max(n_observations, 1)
    )

    (
        logits,
        probabilities,
        y_true,
        attention,
        global_weight,
    ) = predict_loader(
        model,
        loader,
        device,
    )

    y_true = y_true.astype(int)
    predictions = probabilities >= threshold

    metrics = {
        "loss": float(average_loss),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "ks": ks_statistic(
            y_true,
            probabilities,
        ),
        "brier": float(
            brier_score_loss(y_true, probabilities)
        ),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }

    return (
        metrics,
        logits,
        probabilities,
        y_true,
        attention,
        global_weight,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
):

    model.train()

    total_loss = 0.0
    n_observations = 0

    for batch in loader:

        batch = {
            key: (
                {
                    product_name: tensor.to(device)
                    for product_name, tensor in value.items()
                }
                if isinstance(value, dict)
                else value.to(device)
            )
            for key, value in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        output = model(batch)

        loss = criterion(
            output["logits"],
            batch["y"],
        )

        loss.backward()

        nn.utils.clip_grad_norm_(
            model.parameters(),
            CFG.max_grad_norm,
        )

        optimizer.step()

        batch_size = batch["y"].shape[0]

        total_loss += loss.item() * batch_size
        n_observations += batch_size

    return (
        total_loss
        /
        max(n_observations, 1)
    )


# ============================================================
# 16) EARLY STOPPING
# ============================================================


class EarlyStopping:

    def __init__(
        self,
        patience=7,
        min_delta=1e-4,
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_score = None
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def step(
        self,
        score,
        model,
    ):

        if not np.isfinite(score):
            return

        if self.best_score is None:

            self.best_score = score

            self.best_state = copy.deepcopy(
                model.state_dict()
            )

            return

        improved = (
            score
            >
            self.best_score + self.min_delta
        )

        if improved:

            self.best_score = score

            self.best_state = copy.deepcopy(
                model.state_dict()
            )

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True


# ============================================================
# 17) POS WEIGHT
# ============================================================


def compute_pos_weight(y_train):

    y_train = np.asarray(
        y_train,
        dtype=int,
    )

    n_positive = int(
        (y_train == 1).sum()
    )

    n_negative = int(
        (y_train == 0).sum()
    )

    if n_positive == 0 or n_negative == 0:
        raise ValueError(
            "TRAIN debe contener ambas clases."
        )

    return torch.tensor(
        min(n_negative / n_positive, 20.0),
        dtype=torch.float32,
    )


# ============================================================
# 18) GUARDAR ARTEFACTO
# ============================================================


def save_model_artifact(
    model,
    model_config,
    preprocessing,
    threshold,
    validation_metrics,
    test_metrics,
    history,
    path,
):

    model_state_cpu = {
        name: parameter.detach().cpu()
        for name, parameter in model.state_dict().items()
    }

    artifact = {

        "model_state_dict": model_state_cpu,

        "model_config": model_config,

        "preprocessing": preprocessing,

        "feature_names": {
            "global": GLOBAL_FEATURES,
            "products": PRODUCTS,
        },

        "columns": {
            "id": ID_COL,
            "date": DATE_COL,
            "target": TARGET_COL,
        },

        "bucket_values": BUCKET_VALUES,

        "target_definition": {
            "class_0": "no empeora en los proximos 3 meses",
            "class_1": "empeora en los proximos 3 meses",
            "current_state": "bucket actual de cada producto",
        },

        "decision_threshold": float(threshold),

        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "history": history,

        "training_config": asdict(CFG),

        "environment": {
            "pytorch_version": torch.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
    }

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        artifact,
        output_path,
    )

    print(
        "\nModelo guardado:",
        output_path,
    )


# ============================================================
# 19) INFERENCIA SOBRE UNA BASE NUEVA
# ============================================================


@torch.no_grad()
def score_new_base(
    new_df,
    artifact_path,
    device=None,
):

    selected_device = torch.device(
        device
        if device is not None
        else (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    )

    artifact = torch.load(
        artifact_path,
        map_location=selected_device,
        weights_only=False,
    )

    # Las listas guardadas deben coincidir con las listas visibles
    # en este archivo. Esto evita puntuar con columnas distintas.
    if artifact["feature_names"]["global"] != GLOBAL_FEATURES:
        raise ValueError(
            "GLOBAL_FEATURES no coincide con el modelo guardado."
        )

    if artifact["feature_names"]["products"] != PRODUCTS:
        raise ValueError(
            "PRODUCTS no coincide con el modelo guardado."
        )

    audit_base(
        new_df,
        require_target=False,
    )

    (
        X_global_new,
        X_products_new,
        product_mask_new,
        bucket_index_new,
        y_new,
    ) = transform_base(
        new_df,
        artifact["preprocessing"],
        require_target=False,
    )

    new_dataset = CustomerPortfolioDataset(
        X_global_new,
        X_products_new,
        product_mask_new,
        bucket_index_new,
        y_new,
    )

    new_loader = DataLoader(
        new_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    loaded_model = UniversalCollectionsModel(
        **artifact["model_config"]
    ).to(selected_device)

    loaded_model.load_state_dict(
        artifact["model_state_dict"]
    )

    loaded_model.eval()

    (
        _,
        probabilities,
        _,
        attention,
        global_weight,
    ) = predict_loader(
        loaded_model,
        new_loader,
        selected_device,
    )

    result = pd.DataFrame(
        {
            ID_COL: new_df[ID_COL].to_numpy(),
            "risk_score": probabilities,
            "prediction": (
                probabilities
                >=
                artifact["decision_threshold"]
            ).astype(int),
            "weight_global": global_weight,
            "weight_products": 1.0 - global_weight,
        }
    )

    if DATE_COL is not None:
        result[DATE_COL] = new_df[DATE_COL].to_numpy()

    for product_position, product_name in enumerate(PRODUCT_NAMES):

        result[
            f"attention_{product_name}"
        ] = attention[:, product_position]

    return result


# ============================================================
# 20) EJECUCION DEL ENTRENAMIENTO
#
# A partir de aqui el flujo es completamente explicito.
# No existe run_pipeline ni una fabrica escondida.
# ============================================================


if __name__ == "__main__":

    # ========================================================
    # 20.1) CARGA DE DATOS
    # ========================================================

    input_path = Path(INPUT_PATH)

    if input_path.suffix.lower() == ".csv":

        base = pd.read_csv(
            input_path
        )

    elif input_path.suffix.lower() in {".parquet", ".pq"}:

        base = pd.read_parquet(
            input_path
        )

    else:
        raise ValueError(
            "INPUT_PATH debe ser CSV o Parquet."
        )

    # ========================================================
    # 20.2) LIMPIEZA DE FECHA Y AUDITORIA
    # ========================================================

    if DATE_COL is not None:

        base[DATE_COL] = pd.to_datetime(
            base[DATE_COL],
            errors="coerce",
        )

    audit_report = audit_base(
        base,
        require_target=True,
    )

    print(
        "\nAUDITORIA"
    )

    print(
        json.dumps(
            audit_report,
            indent=2,
        )
    )

    # ========================================================
    # 20.3) SPLIT
    # ========================================================

    (
        train_df,
        validation_df,
        test_df,
    ) = split_base(
        base
    )

    print(
        "\nTRAIN:",
        train_df.shape,
    )

    print(
        "VALID:",
        validation_df.shape,
    )

    print(
        "TEST:",
        test_df.shape,
    )

    # ========================================================
    # 20.4) APRENDER PREPROCESAMIENTO EN TRAIN
    # ========================================================

    preprocessing = fit_preprocessing(
        train_df
    )

    # ========================================================
    # 20.5) TRANSFORMAR TRAIN / VALID / TEST
    # ========================================================

    (
        X_global_train,
        X_products_train,
        product_mask_train,
        bucket_index_train,
        y_train,
    ) = transform_base(
        train_df,
        preprocessing,
        require_target=True,
    )

    (
        X_global_validation,
        X_products_validation,
        product_mask_validation,
        bucket_index_validation,
        y_validation,
    ) = transform_base(
        validation_df,
        preprocessing,
        require_target=True,
    )

    (
        X_global_test,
        X_products_test,
        product_mask_test,
        bucket_index_test,
        y_test,
    ) = transform_base(
        test_df,
        preprocessing,
        require_target=True,
    )

    # ========================================================
    # 20.6) DATASETS
    # ========================================================

    train_dataset = CustomerPortfolioDataset(
        X_global_train,
        X_products_train,
        product_mask_train,
        bucket_index_train,
        y_train,
    )

    validation_dataset = CustomerPortfolioDataset(
        X_global_validation,
        X_products_validation,
        product_mask_validation,
        bucket_index_validation,
        y_validation,
    )

    test_dataset = CustomerPortfolioDataset(
        X_global_test,
        X_products_test,
        product_mask_test,
        bucket_index_test,
        y_test,
    )

    # ========================================================
    # 20.7) DATALOADERS
    # ========================================================

    generator = torch.Generator().manual_seed(
        CFG.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=CFG.num_workers,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    # ========================================================
    # 20.8) CONFIGURACION E INSTANCIA DEL MODELO
    # ========================================================

    product_input_dims = {
        product_name: X_products_train[product_name].shape[1]
        for product_name in PRODUCT_NAMES
    }

    model_config = {
        "global_input_dim": X_global_train.shape[1],
        "product_input_dims": product_input_dims,
        "product_names": PRODUCT_NAMES,
        "number_of_bucket_values": len(BUCKET_VALUES),
        "d_model": CFG.d_model,
        "bucket_embedding_dim": CFG.bucket_embedding_dim,
        "product_embedding_dim": CFG.product_embedding_dim,
        "num_heads": CFG.num_heads,
        "attention_layers": CFG.attention_layers,
        "dropout": CFG.dropout,
    }

    model = UniversalCollectionsModel(
        **model_config
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(
        device
    )

    print(
        "\nDevice:",
        device,
    )

    # ========================================================
    # 20.9) LOSS
    # ========================================================

    if CFG.use_pos_weight:

        pos_weight = compute_pos_weight(
            y_train
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        print(
            "pos_weight:",
            pos_weight.item(),
        )

    else:

        criterion = nn.BCEWithLogitsLoss()

        print(
            "BCE sin pos_weight"
        )

    # ========================================================
    # 20.10) OPTIMIZER / SCHEDULER / EARLY STOPPING
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=CFG.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    early_stopper = EarlyStopping(
        patience=CFG.early_stopping_patience,
        min_delta=1e-4,
    )

    history = []

    # ========================================================
    # 20.11) LOOP DE ENTRENAMIENTO
    # ========================================================

    for epoch in range(
        1,
        CFG.num_epochs + 1,
    ):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        (
            train_metrics,
            _,
            _,
            _,
            _,
            _,
        ) = evaluate_model(
            model,
            train_loader,
            criterion,
            device,
        )

        (
            validation_metrics,
            _,
            _,
            _,
            _,
            _,
        ) = evaluate_model(
            model,
            validation_loader,
            criterion,
            device,
        )

        validation_pr_auc = validation_metrics["pr_auc"]

        scheduler.step(
            validation_pr_auc
        )

        early_stopper.step(
            validation_pr_auc,
            model,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_auc": train_metrics["roc_auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "train_ks": train_metrics["ks"],
            "validation_loss": validation_metrics["loss"],
            "validation_auc": validation_metrics["roc_auc"],
            "validation_pr_auc": validation_metrics["pr_auc"],
            "validation_ks": validation_metrics["ks"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"train_auc={train_metrics['roc_auc']:.4f} | "
            f"val_auc={validation_metrics['roc_auc']:.4f} | "
            f"val_pr={validation_metrics['pr_auc']:.4f} | "
            f"val_ks={validation_metrics['ks']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if early_stopper.should_stop:

            print(
                f"\nEarly stopping en epoch {epoch}"
            )

            break

    # ========================================================
    # 20.12) RECUPERAR MEJOR MODELO
    # ========================================================

    if early_stopper.best_state is not None:

        model.load_state_dict(
            early_stopper.best_state
        )

    # ========================================================
    # 20.13) UMBRAL CON VALIDACION
    # ========================================================

    (
        _,
        _,
        validation_probabilities,
        validation_targets,
        _,
        _,
    ) = evaluate_model(
        model,
        validation_loader,
        criterion,
        device,
    )

    threshold_candidates = np.linspace(
        0.05,
        0.95,
        91,
    )

    threshold = max(
        threshold_candidates,
        key=lambda candidate: f1_score(
            validation_targets,
            validation_probabilities >= candidate,
            zero_division=0,
        ),
    )

    # ========================================================
    # 20.14) METRICAS FINALES VALID / TEST
    # ========================================================

    (
        validation_metrics,
        _,
        _,
        _,
        _,
        _,
    ) = evaluate_model(
        model,
        validation_loader,
        criterion,
        device,
        threshold=threshold,
    )

    (
        test_metrics,
        test_logits,
        test_probabilities,
        test_targets,
        test_attention,
        test_global_weight,
    ) = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        threshold=threshold,
    )

    print(
        "\n=============================="
    )

    print(
        "TEST FINAL"
    )

    print(
        "=============================="
    )

    print(
        json.dumps(
            test_metrics,
            indent=2,
        )
    )

    # ========================================================
    # 20.15) SCORES Y ATENCION POR CLIENTE
    # ========================================================

    test_scores = pd.DataFrame(
        {
            ID_COL: test_df[ID_COL].to_numpy(),
            TARGET_COL: test_targets.astype(int),
            "risk_score": test_probabilities,
            "prediction": (
                test_probabilities >= threshold
            ).astype(int),
            "weight_global": test_global_weight,
            "weight_products": 1.0 - test_global_weight,
        }
    )

    if DATE_COL is not None:

        test_scores[DATE_COL] = (
            test_df[DATE_COL].to_numpy()
        )

    for product_position, product_name in enumerate(PRODUCT_NAMES):

        test_scores[
            f"attention_{product_name}"
        ] = test_attention[:, product_position]

    test_scores.to_csv(
        SCORES_PATH,
        index=False,
    )

    print(
        "\nPrimeros scores:"
    )

    print(
        test_scores.head(20)
    )

    # ========================================================
    # 20.16) GUARDAR MODELO
    # ========================================================

    save_model_artifact(
        model=model,
        model_config=model_config,
        preprocessing=preprocessing,
        threshold=threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        history=history,
        path=MODEL_PATH,
    )


# ============================================================
# 21) EJEMPLO DE INFERENCIA
# ============================================================

# nueva_base = pd.read_parquet(
#     "base_clientes_nueva.parquet"
# )
#
# scores_nuevos = score_new_base(
#     new_df=nueva_base,
#     artifact_path=MODEL_PATH,
# )
#
# print(
#     scores_nuevos.head()
# )
