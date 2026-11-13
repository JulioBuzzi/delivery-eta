"""Modelos.

Previsão pontual (quanto tempo vai levar):
  - Mediana global                 : sempre o mesmo número
  - Regra da plataforma            : estimativas de envio + deslocamento + preparo mediano
  - LightGBM (mediana)             : regressão quantílica em 0,5

Prazo prometido ("chega em até X min"):
  - qualquer previsão pontual + uma margem fixa, calibrada pra atingir a pontualidade alvo
  - LightGBM quantílico (0,9) + ajuste conformal: a margem varia por pedido, e o ajuste
    garante a pontualidade alvo nos dados de calibração
"""

import lightgbm as lgb
import numpy as np


# ---------------------------------------------------------------- pontuais
class MedianaGlobal:
    nome = "Mediana global"

    def fit(self, X, y, **_):
        self.m_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m_)


class RegraPlataforma:
    """O que um sistema sem ML faria: soma as estimativas que a plataforma já tem
    (tempo pra loja receber o pedido + deslocamento) e um tempo de preparo mediano."""
    nome = "Regra (envio + deslocamento + preparo mediano)"

    def fit(self, X, y, **_):
        base = X["est_envio_loja"].fillna(0) + X["est_deslocamento"].fillna(X["est_deslocamento"].median())
        self.preparo_ = float(np.median(y - base))
        self.desl_med_ = float(X["est_deslocamento"].median())
        return self

    def predict(self, X):
        return (X["est_envio_loja"].fillna(0) + X["est_deslocamento"].fillna(self.desl_med_) + self.preparo_).to_numpy()


PARAMS_LGB = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=0)


class LGBQuantil:
    """LightGBM com perda quantílica. alpha=0.5 é a mediana (previsão pontual);
    alpha=0.9 é o tempo que só 10% das entregas ultrapassam."""

    def __init__(self, alpha=0.5, params=None, max_rounds=3000):
        self.alpha, self.params, self.max_rounds = alpha, params or PARAMS_LGB, max_rounds
        self.nome = "LightGBM (mediana)" if alpha == 0.5 else f"LightGBM (quantil {alpha:.2f})"

    def fit(self, X, y, X_val=None, y_val=None):
        p = {**self.params, "objective": "quantile", "alpha": self.alpha, "metric": "quantile"}
        dtr = lgb.Dataset(X, y)
        callbacks, valid = [], []
        if X_val is not None:
            valid = [lgb.Dataset(X_val, y_val, reference=dtr)]
            callbacks = [lgb.early_stopping(100, verbose=False)]
        self.booster_ = lgb.train(p, dtr, num_boost_round=self.max_rounds, valid_sets=valid, callbacks=callbacks)
        return self

    def predict(self, X):
        return self.booster_.predict(X, num_iteration=self.booster_.best_iteration or None)


# ----------------------------------------------------------- prazo prometido
def margem_fixa(pred_calib, y_calib, pontualidade):
    """Margem (minutos) que, somada à previsão, deixa `pontualidade` das entregas
    de calibração dentro do prazo."""
    return float(np.quantile(y_calib - pred_calib, pontualidade))


def conformal_online(df, pred_q, pontualidade, janela_h=24, a_cada_h=1, inicio=None):
    """Recalibra o ajuste conformal ao longo do tempo: a cada `a_cada_h` horas, usa só as
    entregas que TERMINARAM nas últimas `janela_h` horas (sem olhar o futuro).
    `df` precisa de created_at, actual_delivery_time e minutos; `pred_q` é o quantil
    previsto pra cada linha de df. Devolve o ajuste (em minutos) pra cada linha a partir de `inicio`."""
    t_pedido = df["created_at"].to_numpy()
    t_fim = df["actual_delivery_time"].to_numpy()
    scores = df["minutos"].to_numpy() - pred_q
    ordem = np.argsort(t_fim)
    t_fim_ord, scores_ord = t_fim[ordem], scores[ordem]

    inicio = np.datetime64(inicio) if inicio is not None else t_pedido.min()
    ajuste = np.full(len(df), np.nan)
    passo = np.timedelta64(a_cada_h, "h")
    t = inicio
    while t <= t_pedido.max():
        a = np.searchsorted(t_fim_ord, t - np.timedelta64(janela_h, "h"), side="left")
        b = np.searchsorted(t_fim_ord, t, side="left")
        s = scores_ord[a:b]
        if len(s) >= 50:
            n = len(s)
            nivel = min(1.0, np.ceil((n + 1) * pontualidade) / n)
            valor = float(np.quantile(s, nivel, method="higher"))
        else:
            valor = np.nan
        alvo = (t_pedido >= t) & (t_pedido < t + passo)
        ajuste[alvo] = valor
        t = t + passo
    return ajuste


def ajuste_conformal(pred_q_calib, y_calib, pontualidade):
    """Conformal (split): quanto somar ao quantil previsto pra que a fração de entregas
    no prazo seja pelo menos `pontualidade`. Com n pontos de calibração, a garantia
    vale em média pra dados que se comportam como os de calibração."""
    scores = y_calib - pred_q_calib
    n = len(scores)
    nivel = min(1.0, np.ceil((n + 1) * pontualidade) / n)
    return float(np.quantile(scores, nivel, method="higher"))