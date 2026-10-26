# =====================================================================
# TRABALHO PRÁTICO: Tópicos Avançados em Inteligência Artificial
# Projeto: Classificação de Vulnerabilidade Social (BPC / CadÚnico)
# Grupo: Kaique Fernando, Otavio Carvalho e Rhayssa Ramos
#
# Código de treinamento (preparação, treino, comparação e avaliação)
#
# NOTA DE TRANSPARÊNCIA: código desenvolvido com auxílio de IA (LLM),
# revisado e validado pelos integrantes do grupo.
# =====================================================================
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ATRIBUTOS = ["renda_per_capita", "qnt_pessoas", "tem_pcd_ou_idoso", "moradia_precaria"]
ALVO = "alta_vulnerabilidade"
SEMENTE = 42
relatorio = []  # guarda as saídas pra salvar em resultados/metricas.txt


def log(texto=""):
    print(texto)
    relatorio.append(str(texto))


# 1. CARREGAMENTO E ANÁLISE INICIAL DOS DADOS
df = pd.read_csv("dados/dados_vulnerabilidade_sintetico.csv")
log("=== 1. ANÁLISE INICIAL ===")
log(f"Registros: {len(df)} | Colunas: {list(df.columns)}")
log(f"Valores ausentes por coluna:\n{df.isna().sum()}")
log(f"Cadastros duplicados (mesmo cod_familia): {df.duplicated().sum()}")
log(f"Distribuição da classe-alvo:\n{df[ALVO].value_counts()}")

# 2. LIMPEZA
# remover duplicatas não depende do treino, pode ser antes da divisão
df = df.drop_duplicates().drop(columns=["cod_familia"])
log(f"\nApós remover duplicatas: {len(df)} registros")

X = df[ATRIBUTOS]
y = df[ALVO]

# 3. DIVISÃO EM TREINO E TESTE (80/20, estratificada)
# o teste só é usado no final
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEMENTE)
log(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

# 4. MODELOS COMPARADOS
# imputação e escala ficam dentro do Pipeline pra serem ajustadas só no treino
alternativas = {
    "Regressão Logística": Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escala", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=1000, random_state=SEMENTE)),
    ]),
    "Árvore de Decisão (prof. 4)": Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("modelo", DecisionTreeClassifier(max_depth=4, min_samples_leaf=10,
                                          random_state=SEMENTE)),
    ]),
}

# 5. VALIDAÇÃO CRUZADA NO TREINO (5 folds)
# critério de escolha: F1 da classe 1
log("\n=== 2. VALIDAÇÃO CRUZADA NO TREINO (5 folds) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
resumo_cv = {}
for nome, pipe in alternativas.items():
    r = cross_validate(pipe, X_train, y_train, cv=cv,
                       scoring={"f1": "f1", "recall": "recall", "acuracia": "accuracy"})
    resumo_cv[nome] = {k: round(float(r[f"test_{k}"].mean()), 3) for k in ["f1", "recall", "acuracia"]}
    resumo_cv[nome]["f1_desvio"] = round(float(r["test_f1"].std()), 3)
    log(f"{nome}: {resumo_cv[nome]}")

escolhido = max(resumo_cv, key=lambda n: resumo_cv[n]["f1"])
diferenca = abs(resumo_cv["Regressão Logística"]["f1"] - resumo_cv["Árvore de Decisão (prof. 4)"]["f1"])
log(f"\nMaior F1 na validação cruzada: {escolhido} (diferença de F1 entre as duas: {diferenca:.3f})")

# 6. AVALIAÇÃO NO CONJUNTO DE TESTE
log("\n=== 3. AVALIAÇÃO NO CONJUNTO DE TESTE ===")
nomes_classes = ["Baixa/Moderada (0)", "Alta (1)"]
fig, eixos = plt.subplots(1, 2, figsize=(10, 4))
metricas_teste = {}
for eixo, (nome, pipe) in zip(eixos, alternativas.items()):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    log(f"\n--- {nome} ---")
    log(classification_report(y_test, pred, target_names=nomes_classes, digits=3))
    cm = confusion_matrix(y_test, pred)
    log(f"Matriz de confusão [linhas = real, colunas = previsto]:\n{cm}")
    log(f"Falsos negativos (família vulnerável classificada como regular): {cm[1, 0]}")
    log(f"Falsos positivos (família regular classificada como vulnerável): {cm[0, 1]}")
    metricas_teste[nome] = {
        "f1_classe1": round(float(f1_score(y_test, pred)), 3),
        "recall_classe1": round(float(recall_score(y_test, pred)), 3),
        "matriz_confusao": cm.tolist(),
    }
    ConfusionMatrixDisplay(cm, display_labels=nomes_classes).plot(ax=eixo, colorbar=False, cmap="Blues")
    eixo.set_title(nome)
    eixo.set_xlabel("Classe prevista")
    eixo.set_ylabel("Classe real")
plt.tight_layout()
plt.savefig("resultados/matrizes_confusao.png", dpi=150)

# 7. INTERPRETAÇÃO
log("\n=== 4. INTERPRETAÇÃO ===")
coefs = alternativas["Regressão Logística"].named_steps["modelo"].coef_[0]
log("Coeficientes da Regressão Logística (dados padronizados):")
for a, c in sorted(zip(ATRIBUTOS, coefs), key=lambda t: -abs(t[1])):
    log(f"  {a:20s} {c:+.3f}")

arvore = alternativas["Árvore de Decisão (prof. 4)"].named_steps["modelo"]
log("\nImportância dos atributos na Árvore:")
for a, imp in sorted(zip(ATRIBUTOS, arvore.feature_importances_), key=lambda t: -t[1]):
    log(f"  {a:20s} {imp:.3f}")
log("\nRegras aprendidas pela Árvore:")
log(export_text(arvore, feature_names=ATRIBUTOS))

plt.figure(figsize=(18, 8))
plot_tree(arvore, feature_names=ATRIBUTOS, class_names=["Regular", "Alta"],
          filled=True, rounded=True, fontsize=8)
plt.savefig("resultados/arvore_decisao.png", dpi=120, bbox_inches="tight")

# 8. SALVANDO O MODELO ESCOLHIDO
MODELO_DEPLOY = escolhido
joblib.dump(alternativas[MODELO_DEPLOY], "modelo/modelo_vulnerabilidade.pkl")
log(f"\nModelo salvo para o deploy: {MODELO_DEPLOY} -> modelo/modelo_vulnerabilidade.pkl")

with open("resultados/metricas.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(relatorio))
with open("resultados/metricas.json", "w", encoding="utf-8") as f:
    json.dump({"validacao_cruzada": resumo_cv, "teste": metricas_teste,
               "modelo_deploy": MODELO_DEPLOY}, f, ensure_ascii=False, indent=2)
