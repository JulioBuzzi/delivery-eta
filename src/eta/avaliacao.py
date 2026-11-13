"""Avaliação do prazo prometido.

Pra cada pontualidade alvo (ex.: 90% das entregas dentro do prazo), cada método escolhe
seu prazo usando só treino + calibração, e mede no teste:
  - pontualidade real      : % de entregas que chegaram dentro do prazo prometido
  - prazo médio prometido  : quanto o cliente vê no app (menor = app mais atraente)
  - atraso > 10 min        : % de entregas que passaram do prazo em mais de 10 minutos
"""

import numpy as np
import pandas as pd

from .models import LGBQuantil, MedianaGlobal, RegraPlataforma, ajuste_conformal, margem_fixa

ALVOS = (0.80, 0.85, 0.90, 0.95)


def metricas_prazo(prazo, y):
    atraso = y - prazo
    return {
        "pontualidade": float(np.mean(atraso <= 0)),
        "prazo_medio": float(np.mean(prazo)),
        "atraso_10min": float(np.mean(atraso > 10)),
        "folga_media": float(np.mean(np.clip(-atraso, 0, None))),  # quanto chegou antes do prometido
    }


def treinar_tudo(X, y, P, alvos=ALVOS):
    """Treina os modelos pontuais e os quantílicos (um por alvo). P = máscaras dos períodos."""
    tr, va = P["treino"], P["validacao"]
    pontuais = [MedianaGlobal().fit(X[tr], y[tr]), RegraPlataforma().fit(X[tr], y[tr]),
                LGBQuantil(0.5).fit(X[tr], y[tr], X[va], y[va])]
    quantis = {a: LGBQuantil(a).fit(X[tr], y[tr], X[va], y[va]) for a in alvos}
    return pontuais, quantis


def prazos(pontuais, quantis, X, y, P, alvos=ALVOS, periodo="teste"):
    """Devolve {(método, alvo): prazo prometido pra cada pedido do período}."""
    ca, te = P["calibracao"], P[periodo]
    out = {}
    for alvo in alvos:
        for m in pontuais:
            pc = m.predict(X[ca])
            out[(m.nome + " + margem fixa", alvo)] = m.predict(X[te]) + margem_fixa(pc, y[ca], alvo)
        q = quantis[alvo]
        ajuste = ajuste_conformal(q.predict(X[ca]), y[ca], alvo)
        out[(f"LightGBM quantílico + conformal", alvo)] = q.predict(X[te]) + ajuste
    return out


def curva_tradeoff(pred, y, pontualidades=np.arange(0.70, 0.981, 0.01)):
    """Fronteira pontualidade x prazo médio: pra cada pontualidade, a margem constante que
    o método precisaria somar À SUA previsão pra atingi-la exatamente nesses dados.
    Serve pra comparar a eficiência dos métodos (quem promete menos pra mesma pontualidade),
    não pra escolher a margem (isso é feito na calibração)."""
    residuo = y - pred
    linhas = []
    for p in pontualidades:
        m = np.quantile(residuo, p)
        linhas.append({"pontualidade": float(p), "prazo_medio": float(np.mean(pred + m))})
    return pd.DataFrame(linhas)


def tabela(prazos_dict, y_teste):
    linhas = [{"metodo": m, "alvo": a, **metricas_prazo(p, y_teste)} for (m, a), p in prazos_dict.items()]
    return pd.DataFrame(linhas)