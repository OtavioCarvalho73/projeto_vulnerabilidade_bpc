# =====================================================================
# TRABALHO PRÁTICO: Tópicos Avançados em Inteligência Artificial
# Geração do dataset sintético
# Grupo: Otavio Carvalho, Kaique Fernando e Rhayssa Ramos
#
# Os microdados identificados do CadÚnico não são públicos e a base
# desidentificada não tem um rótulo de vulnerabilidade, então geramos
# uma base sintética com distribuições parecidas com o perfil do CadÚnico.
# O rótulo vem de um índice de vulnerabilidade + ruído.
#
# NOTA DE TRANSPARÊNCIA: código desenvolvido com auxílio de IA (LLM),
# revisado e validado pelos integrantes do grupo.
# =====================================================================
import numpy as np
import pandas as pd

SEMENTE = 2026
N = 2000
rng = np.random.default_rng(SEMENTE)

# Quantidade de pessoas no domicílio (1 a 10)
qnt_pessoas = np.clip(rng.poisson(2.6, N) + 1, 1, 10)

# Renda per capita (R$): parte sem renda, parte em extrema pobreza e o resto baixa renda
grupo = rng.random(N)
renda = np.where(
    grupo < 0.12, 0.0,                                   # sem renda declarada
    np.where(grupo < 0.45, rng.uniform(20, 220, N),      # extrema pobreza / pobreza
             rng.lognormal(mean=6.1, sigma=0.45, size=N)))  # baixa renda
renda = np.round(np.clip(renda, 0, 1500), 2)

# Presença de idoso ou PcD no domicílio (mais provável em casas maiores)
tem_pcd_ou_idoso = (rng.random(N) < 0.25 + 0.03 * (qnt_pessoas - 1)).astype(int)

# Moradia precária (mais provável quanto menor a renda)
p_moradia = 0.55 - 0.45 * np.clip(renda / 800, 0, 1)
moradia_precaria = (rng.random(N) < p_moradia).astype(int)

# ---------------- Rótulo (variável-alvo) ----------------
# Índice: renda baixa pesa mais, os outros fatores somam pontos.
# O ruído representa a avaliação do técnico, que não é exata.
indice = (
    3.0 * (1 - np.clip(renda / 400, 0, 1))      # renda (0 a 3 pontos)
    + 0.25 * np.clip(qnt_pessoas - 3, 0, 5)     # adensamento (0 a 1,25)
    + 1.0 * tem_pcd_ou_idoso
    + 1.2 * moradia_precaria
    + rng.normal(0, 0.6, N)                     # ruído
)
alta_vulnerabilidade = (indice >= 2.6).astype(int)

df = pd.DataFrame({
    "cod_familia": [f"F{i:05d}" for i in range(1, N + 1)],  # código fictício
    "renda_per_capita": renda,
    "qnt_pessoas": qnt_pessoas.astype(float),
    "tem_pcd_ou_idoso": tem_pcd_ou_idoso.astype(float),
    "moradia_precaria": moradia_precaria.astype(float),
    "alta_vulnerabilidade": alta_vulnerabilidade,
})

# Valores ausentes: ~3% na renda e ~2% na moradia
df.loc[rng.random(N) < 0.03, "renda_per_capita"] = np.nan
df.loc[rng.random(N) < 0.02, "moradia_precaria"] = np.nan
# 40 cadastros duplicados
dup = df.sample(40, random_state=SEMENTE)
df = pd.concat([df, dup], ignore_index=True).sample(frac=1, random_state=SEMENTE)

df.to_csv("dados/dados_vulnerabilidade_sintetico.csv", index=False)
print(f"Dataset gerado: {len(df)} linhas")
print(df["alta_vulnerabilidade"].value_counts(normalize=True).round(3))
