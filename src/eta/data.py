"""Dados: 197 mil entregas do DoorDash (21/01 a 18/02/2015), publicadas pela empresa
como teste técnico ("DoorDash ETA Prediction", também no Kaggle).

O alvo é o tempo total de entrega, em minutos: do pedido feito (created_at) até a
entrega (actual_delivery_time). Os horários estão em UTC; o dataset não informa o fuso
de cada mercado.

Divisão temporal (nada do futuro entra no treino):
  treino     21/01 – 03/02   ajustar os modelos
  validação  04/02 – 08/02   early stopping e escolha de hiperparâmetros
  calibração 09/02 – 11/02   ajuste conformal da margem do prazo prometido
  teste      12/02 – 18/02   avaliação final (só olhado no fim)
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "historical_data.csv"

# cópia pública do arquivo original, com commit fixo; o md5 garante que é o mesmo arquivo
FONTE = ("https://raw.githubusercontent.com/AR10129/Delivery-Duration-Prediction/"
         "da5c29e19716739603721720a5477fae07a0c297/dataset/historical_data.csv")
MD5 = "eab96e5908906bd46743381c477d44e0"

CORTES = {
    "treino": ("2015-01-21", "2015-02-04"),
    "validacao": ("2015-02-04", "2015-02-09"),
    "calibracao": ("2015-02-09", "2015-02-12"),
    "teste": ("2015-02-12", "2015-02-19"),
}

# entregas fora dessa faixa são erros de registro (ex.: 141.947 min = 98 dias)
MIN_MINUTOS, MAX_MINUTOS = 5, 180


def download(path=RAW):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("baixando historical_data.csv (19 MB)...")
        urllib.request.urlretrieve(FONTE, path)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != MD5:
        raise ValueError(f"historical_data.csv com md5 {md5}, esperado {MD5}. Baixe de novo.")
    return path


def load(path=RAW, verbose=False):
    """Lê, cria o alvo em minutos, remove registros inválidos e marca o período de cada linha."""
    df = pd.read_csv(download(path))
    n0 = len(df)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])
    df = df.dropna(subset=["actual_delivery_time"])
    df["minutos"] = (df["actual_delivery_time"] - df["created_at"]).dt.total_seconds() / 60

    fora_periodo = df["created_at"] < CORTES["treino"][0]
    fora_faixa = (df["minutos"] < MIN_MINUTOS) | (df["minutos"] > MAX_MINUTOS)
    df = df[~fora_periodo & ~fora_faixa].copy()

    df["periodo"] = None
    for nome, (ini, fim) in CORTES.items():
        df.loc[(df["created_at"] >= ini) & (df["created_at"] < fim), "periodo"] = nome

    df = df.sort_values("created_at").reset_index(drop=True)
    df["id"] = np.arange(len(df))
    if verbose:
        print(f"{n0} linhas -> {len(df)} entregas válidas "
              f"({n0 - len(df)} removidas: sem horário de entrega, fora do período ou fora de {MIN_MINUTOS}–{MAX_MINUTOS} min)")
    return df