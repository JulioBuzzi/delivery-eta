"""Features. Tudo o que depende de resultado de entrega usa só entregas que já tinham
TERMINADO no momento do pedido (actual_delivery_time < created_at). É o que a plataforma
saberia de verdade na hora de mostrar o prazo.
"""

import numpy as np
import pandas as pd

CATEGORICAS = ["market_id", "order_protocol", "categoria"]


def _historico_por_grupo(df, chave, janela_min=None, prefixo="", k_suavizacao=None, media_global=None):
    """Pra cada pedido: quantas entregas do mesmo grupo (loja ou mercado) já tinham terminado
    e o tempo médio delas. Com `janela_min`, só as que terminaram nos últimos N minutos."""
    cont = np.zeros(len(df))
    soma = np.zeros(len(df))
    for _, g in df.groupby(chave, sort=False):
        pedidos_t = g["created_at"].to_numpy()
        ordem = np.argsort(g["actual_delivery_time"].to_numpy())
        fim_t = g["actual_delivery_time"].to_numpy()[ordem]
        dur = g["minutos"].to_numpy()[ordem]
        cum = np.concatenate([[0.0], np.cumsum(dur)])
        # quantas entregas terminaram antes de cada pedido
        ate = np.searchsorted(fim_t, pedidos_t, side="left")
        if janela_min is None:
            desde = np.zeros_like(ate)
        else:
            desde = np.searchsorted(fim_t, pedidos_t - np.timedelta64(janela_min, "m"), side="left")
        cont[g.index] = ate - desde
        soma[g.index] = cum[ate] - cum[desde]

    out = pd.DataFrame(index=df.index)
    out[f"{prefixo}n"] = cont
    if k_suavizacao is not None:
        # média "puxada" pra média global quando há poucas entregas (loja nova)
        out[f"{prefixo}media"] = (soma + k_suavizacao * media_global) / (cont + k_suavizacao)
    else:
        out[f"{prefixo}media"] = np.where(cont > 0, soma / np.maximum(cont, 1), np.nan)
    return out


def build(df, media_global):
    """Monta a matriz de features. `media_global` deve vir só do treino."""
    X = pd.DataFrame(index=df.index)

    # pedido
    X["itens"] = df["total_items"]
    X["itens_distintos"] = df["num_distinct_items"]
    X["subtotal"] = df["subtotal"] / 100
    X["preco_min"] = df["min_item_price"] / 100
    X["preco_max"] = df["max_item_price"] / 100

    # loja e mercado
    X["market_id"] = df["market_id"].fillna(-1).astype(int)
    X["order_protocol"] = df["order_protocol"].fillna(-1).astype(int)
    cat = df["store_primary_category"].fillna("desconhecida")
    X["categoria"] = cat

    # horário (UTC)
    X["hora"] = df["created_at"].dt.hour
    X["dia_semana"] = df["created_at"].dt.dayofweek

    # oferta x demanda de entregadores no momento do pedido
    on, busy, fila = df["total_onshift_dashers"], df["total_busy_dashers"], df["total_outstanding_orders"]
    X["entregadores_online"] = on
    X["entregadores_ocupados"] = busy
    X["pedidos_na_fila"] = fila
    X["entregadores_livres"] = on - busy
    X["fila_por_entregador"] = fila / on.where(on > 0)
    X["ocupacao"] = busy / on.where(on > 0)

    # estimativas da própria plataforma (em minutos)
    X["est_envio_loja"] = df["estimated_order_place_duration"] / 60
    X["est_deslocamento"] = df["estimated_store_to_consumer_driving_duration"] / 60

    # histórico: lojas (desde o começo) e mercado (última hora), só entregas já concluídas
    loja = _historico_por_grupo(df, "store_id", prefixo="loja_", k_suavizacao=10, media_global=media_global)
    mercado = _historico_por_grupo(df.assign(market_id=X["market_id"]), "market_id", janela_min=60, prefixo="mercado_1h_")
    X = pd.concat([X, loja, mercado], axis=1)
    X["loja_media_menos_deslocamento"] = X["loja_media"] - X["est_deslocamento"]

    for c in CATEGORICAS:
        X[c] = X[c].astype("category")
    return X