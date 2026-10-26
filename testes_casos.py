# =====================================================================
# Casos de teste do relatório (mesmas funções usadas no app.py)
# Saída: resultados/casos_teste.csv
# =====================================================================
import pandas as pd

from inferencia import carregar_modelo, classificar, validar_entrada

CASOS = [
    # (id, tipo, renda, pessoas, pcd_idoso, moradia, esperado)
    ("1", "Válido - extrema pobreza", 80.0, 5, 1, 1, "Alta vulnerabilidade"),
    ("2", "Válido - renda mais alta", 600.0, 2, 0, 0, "Baixa/moderada"),
    ("3", "Válido - renda média com fatores de risco", 300.0, 3, 1, 1, "Alta vulnerabilidade"),
    ("4a", "Inválido - renda negativa", -100.0, 3, 0, 0, "Mensagem de erro, sem classificar"),
    ("4b", "Incompleto - renda e moradia em branco", None, 3, 0, None, "Mensagem de erro, sem classificar"),
    ("5", "Limítrofe - sem renda, mora sozinho", 0.0, 1, 0, 0, "Sem resultado real conhecido (pela regra do dataset, ~75% de chance de alta)"),
]

modelo = carregar_modelo()
linhas = []
for cid, tipo, renda, pessoas, pcd, moradia, esperado in CASOS:
    erros = validar_entrada(renda, pessoas, pcd, moradia)
    if erros:
        obtido = "ERRO: " + " | ".join(erros)
    else:
        classe, prob = classificar(modelo, renda, pessoas, pcd, moradia)
        rotulo = "Alta vulnerabilidade" if classe == 1 else "Baixa/moderada"
        obtido = f"{rotulo} (prob. alta = {prob:.0%})"
    linhas.append({"caso": cid, "tipo": tipo, "renda": renda, "pessoas": pessoas,
                   "pcd_idoso": pcd, "moradia_precaria": moradia,
                   "esperado": esperado, "obtido": obtido})
    print(f"Caso {cid} [{tipo}]\n  entrada: renda={renda}, pessoas={pessoas}, pcd/idoso={pcd}, "
          f"moradia={moradia}\n  esperado: {esperado}\n  obtido:   {obtido}\n")

pd.DataFrame(linhas).to_csv("resultados/casos_teste.csv", index=False)
