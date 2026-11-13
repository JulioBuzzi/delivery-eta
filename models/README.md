---
language: pt
license: mit
library_name: lightgbm
tags:
  - delivery
  - eta
  - quantile-regression
  - conformal-prediction
  - lightgbm
  - tabular-regression
metrics:
  - mae
pipeline_tag: tabular-regression
---

# delivery-eta: prazo de entrega com LightGBM quantílico + conformal

Modelos que preveem o tempo de entrega de pedidos de comida e definem o **prazo prometido** ao cliente, com a meta de 90% das entregas dentro do prazo.

- **Código, notebooks e avaliação:** [github.com/JulioBuzzi/delivery-eta](https://github.com/JulioBuzzi/delivery-eta)
- **Demo:** [Space no Hugging Face](https://huggingface.co/spaces/JulioBuzzi/delivery-eta)

## Arquivos

| Arquivo | O que é |
|---|---|
| `lgb_q50.txt` | LightGBM, quantil 0,5: tempo previsto (mediana) |
| `lgb_q90.txt` | LightGBM, quantil 0,9: base do prazo prometido |
| `config.json` | lista de features, categorias, ajuste conformal inicial e métricas de teste |

O prazo prometido é `q90(pedido) + ajuste`. O ajuste conformal é recalculado a cada hora, com as entregas concluídas nas últimas 24h (veja `conformal_online` no repositório).

```python
import lightgbm as lgb
q90 = lgb.Booster(model_file="lgb_q90.txt")
# X: DataFrame com as features de config.json (gerado por eta.features.build)
prazo = q90.predict(X) + ajuste
```

## Avaliação

Semana de teste: 12 a 18/02/2015, 50.210 entregas, sem nenhuma decisão tomada com esses dados.

| Método | Pontualidade (meta 90%) | Prazo médio | Atraso > 10 min | No prazo, pior faixa de pedido |
|---|---|---|---|---|
| Regra (estimativas da plataforma + margem) | 88,7% | 69,8 min | 5,6% | 70,6% |
| LightGBM mediana + margem | 89,8% | 65,9 min | 4,4% | 86,5% |
| **LightGBM quantílico + conformal (este modelo)** | **89,8%** | **66,4 min** | **4,1%** | **89,1%** |

O MAE da mediana é de 10,3 min, contra 12,9 min da regra.

## Dados

*DoorDash ETA Prediction*: teste técnico público do DoorDash, com 197 mil entregas de 21/01 a 18/02/2015 em 6 mercados dos EUA e ruído adicionado pela empresa.

## Limitações

- Treinado com dados de 2015 de uma única plataforma. Use como referência de método, não pra produção.
- A garantia de pontualidade é em média e pressupõe que as próximas horas se pareçam com as últimas 24h.