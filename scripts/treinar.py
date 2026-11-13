"""Treina os modelos finais, salva em models/ e gera os dados da demo (static/demo.json).

    python scripts/treinar.py

- models/lgb_q50.txt e models/lgb_q90.txt : LightGBM (mediana e quantil 0,9)
- models/config.json                     : features, categorias, ajuste conformal, métricas
- static/demo.json                       : resultados da semana de teste pra página da demo
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eta import build, load  # noqa: E402
from eta.avaliacao import ALVOS, metricas_prazo, treinar_tudo  # noqa: E402
from eta.models import ajuste_conformal, conformal_online  # noqa: E402

df = load(verbose=True)
media_treino = float(df.loc[df["periodo"] == "treino", "minutos"].mean())
X = build(df, media_treino)
y = df["minutos"].to_numpy()
P = {p: (df["periodo"] == p).to_numpy() for p in ["treino", "validacao", "calibracao", "teste"]}
te = P["teste"]

pontuais, quantis = treinar_tudo(X, y, P)
regra, lgb50 = pontuais[1], pontuais[2]
print("modelos treinados")


def online(pred, alvo):
    fixo = ajuste_conformal(pred[P["calibracao"]], y[P["calibracao"]], alvo)
    aj = conformal_online(df, pred, alvo, janela_h=24, a_cada_h=1, inicio="2015-02-09")
    return np.where(np.isnan(aj), fixo, aj), fixo


# ------------------------------------------------------------------ modelos
pasta = ROOT / "models"
pasta.mkdir(exist_ok=True)
lgb50.booster_.save_model(str(pasta / "lgb_q50.txt"), num_iteration=lgb50.booster_.best_iteration)
quantis[0.9].booster_.save_model(str(pasta / "lgb_q90.txt"), num_iteration=quantis[0.9].booster_.best_iteration)

p50 = lgb50.predict(X)
p90 = quantis[0.9].predict(X)
aj90, fixo90 = online(p90, 0.9)
config = {
    "features": list(X.columns),
    "categorias": {c: [str(v) for v in X[c].cat.categories] for c in ["market_id", "order_protocol", "categoria"]},
    "media_treino_min": media_treino,
    "ajuste_conformal_fixo_q90_min": fixo90,
    "recalibracao_online": {"janela_h": 24, "a_cada_h": 1},
    "metricas_teste": {
        "mae_lgb_q50": float(np.mean(np.abs(y[te] - p50[te]))),
        "mae_regra": float(np.mean(np.abs(y[te] - regra.predict(X[te])))),
        "q90_conformal_online": metricas_prazo(p90[te] + aj90[te], y[te]),
    },
}
(pasta / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
print("-> models/ (lgb_q50.txt, lgb_q90.txt, config.json)")

# --------------------------------------------------------------------- demo
faixa = pd.qcut(p50[te], 5, labels=False)
nomes_faixa = ["muito rápido", "rápido", "médio", "lento", "muito lento"]
metodos = {}
for alvo in ALVOS:
    preds = {
        "regra": regra.predict(X),
        "mediana": p50,
        "quantil": quantis[alvo].predict(X),
    }
    for chave, pred in preds.items():
        aj, _ = online(pred, alvo)
        prazo = pred[te] + aj[te]
        por_f = pd.Series(y[te] <= prazo).groupby(faixa).mean()
        metodos.setdefault(chave, {})[f"{alvo:.2f}"] = {
            **{k: round(v, 4) for k, v in metricas_prazo(prazo, y[te]).items()},
            "por_faixa": [round(float(v), 4) for v in por_f.values],
        }

# pedidos de exemplo: amostra da semana de teste com o prazo que cada método mostraria (meta 90%)
pr_regra = regra.predict(X)
aj_regra, _ = online(pr_regra, 0.9)
rng = np.random.default_rng(0)
idx_teste = np.where(te)[0]
amostra = rng.choice(idx_teste, 300, replace=False)
exemplos = []
for i in sorted(amostra, key=lambda k: df.loc[k, "created_at"]):
    r = df.loc[i]
    prazo = p90[i] + aj90[i]
    exemplos.append({
        "quando": r["created_at"].strftime("%d/%m %H:%M") + " UTC",
        "mercado": int(r["market_id"]) if pd.notna(r["market_id"]) else None,
        "categoria": str(r["store_primary_category"]) if pd.notna(r["store_primary_category"]) else "desconhecida",
        "itens": int(r["total_items"]),
        "subtotal": round(r["subtotal"] / 100, 2),
        "fila_por_entregador": None if pd.isna(X.loc[i, "fila_por_entregador"]) else round(float(X.loc[i, "fila_por_entregador"]), 2),
        "deslocamento_min": None if pd.isna(X.loc[i, "est_deslocamento"]) else round(float(X.loc[i, "est_deslocamento"]), 1),
        "loja_entregas_antes": int(X.loc[i, "loja_n"]),
        "previsto_mediana": round(float(p50[i]), 1),
        "prazo": round(float(prazo), 1),
        "regra_prazo": round(float(pr_regra[i] + aj_regra[i]), 1),
        "real": round(float(y[i]), 1),
    })

demo = {"alvos": [f"{a:.2f}" for a in ALVOS], "faixas": nomes_faixa, "metodos": metodos,
        "n_teste": int(te.sum()), "exemplos": exemplos}
(ROOT / "static").mkdir(exist_ok=True)
(ROOT / "static" / "demo.json").write_text(json.dumps(demo, ensure_ascii=False))
print("-> static/demo.json")