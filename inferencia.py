# =====================================================================
# Validação e inferência, usadas pelo app.py e pelo testes_casos.py
# (assim os testes locais e o app publicado usam o mesmo código)
#
# NOTA DE TRANSPARÊNCIA: código desenvolvido com auxílio de IA (LLM),
# revisado e validado pelos integrantes do grupo.
# =====================================================================
from pathlib import Path

import joblib
import pandas as pd

ATRIBUTOS = ["renda_per_capita", "qnt_pessoas", "tem_pcd_ou_idoso", "moradia_precaria"]
CAMINHO_MODELO = Path(__file__).parent / "modelo" / "modelo_vulnerabilidade.pkl"

RENDA_MAX = 10000.0   # acima disso provavelmente é erro de digitação
PESSOAS_MAX = 20


def carregar_modelo(caminho=CAMINHO_MODELO):
    # carrega o pipeline salvo (imputador + escala + modelo)
    return joblib.load(caminho)


def validar_entrada(renda, pessoas, pcd_idoso, moradia):
    # retorna a lista de erros (vazia = entrada válida)
    erros = []
    if renda is None:
        erros.append("Informe a renda per capita (pode ser 0 se a família não tem renda).")
    elif renda < 0:
        erros.append("A renda per capita não pode ser negativa.")
    elif renda > RENDA_MAX:
        erros.append("Renda per capita acima de R$ 10.000; confira se não houve erro de digitação.")

    if pessoas is None:
        erros.append("Informe a quantidade de pessoas no domicílio.")
    elif pessoas != int(pessoas) or not 1 <= pessoas <= PESSOAS_MAX:
        erros.append(f"A quantidade de pessoas deve ser um número inteiro entre 1 e {PESSOAS_MAX}.")

    if pcd_idoso not in (0, 1):
        erros.append("Indique se há pessoa idosa ou com deficiência no domicílio.")
    if moradia not in (0, 1):
        erros.append("Indique se a moradia é precária.")
    return erros


def classificar(modelo, renda, pessoas, pcd_idoso, moradia):
    # faz uma inferência e retorna (classe, probabilidade de alta vulnerabilidade)
    entrada = pd.DataFrame([[float(renda), float(pessoas), float(pcd_idoso), float(moradia)]],
                           columns=ATRIBUTOS)
    classe = int(modelo.predict(entrada)[0])
    prob_alta = float(modelo.predict_proba(entrada)[0][1])
    return classe, prob_alta
