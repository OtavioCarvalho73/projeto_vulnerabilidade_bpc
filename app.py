# =====================================================================
# TRABALHO PRÁTICO: Tópicos Avançados em Inteligência Artificial
# Código da Interface e Deploy (Streamlit)
# Grupo: Otavio Carvalho, Kaique Fernando e Rhayssa Ramos
#
# NOTA DE TRANSPARÊNCIA: código desenvolvido com auxílio de IA (LLM),
# revisado e validado pelos integrantes do grupo.
# =====================================================================
import streamlit as st

from inferencia import carregar_modelo, classificar, validar_entrada

st.set_page_config(page_title="Vigilância Socioassistencial - BPC", layout="centered")


@st.cache_resource
def obter_modelo():
    # carrega o modelo uma vez só, cada clique faz apenas a previsão
    return carregar_modelo()


st.title("Classificador de Vulnerabilidade Social")
st.write(
    "Aplicação para apoiar técnicos da Vigilância Socioassistencial (SUAS/CRAS) na triagem "
    "de famílias. Com base em quatro informações do cadastro, o modelo indica se a família "
    "tende a estar em alta vulnerabilidade (prioridade de acompanhamento) ou em "
    "vulnerabilidade baixa/moderada."
)
st.warning(
    "Trabalho acadêmico: o modelo foi treinado com dados sintéticos, não com dados reais "
    "do CadÚnico. O resultado não substitui a avaliação do técnico de referência."
)

try:
    modelo = obter_modelo()
except Exception as erro:
    st.error(f"Erro ao carregar o modelo treinado: {erro}")
    st.stop()

st.markdown("---")
st.subheader("Insira os dados da família:")

renda = st.number_input(
    "Renda per capita (R$ por mês)",
    value=None, step=10.0, format="%.2f", placeholder="Ex.: 150.00",
    help="Renda mensal de todos os moradores dividida pelo número de moradores. "
         "Use 0 se a família não tem renda.",
)
pessoas = st.number_input(
    "Quantidade de pessoas no domicílio",
    value=None, step=1, placeholder="Ex.: 4",
    help="Total de moradores, contando crianças (de 1 a 20).",
)
pcd_idoso = st.selectbox(
    "Há pessoa idosa (65+) ou com deficiência no domicílio?",
    options=[1, 0], index=None, placeholder="Selecione",
    format_func=lambda x: "Sim" if x == 1 else "Não",
)
moradia = st.selectbox(
    "A moradia é precária?",
    options=[1, 0], index=None, placeholder="Selecione",
    format_func=lambda x: "Sim" if x == 1 else "Não",
    help="Sim: sem acesso adequado a água encanada e/ou esgoto.",
)

st.markdown("---")

if st.button("Classificar vulnerabilidade", type="primary"):
    erros = validar_entrada(renda, pessoas, pcd_idoso, moradia)
    if erros:
        st.error("Não foi possível classificar. Verifique os campos:")
        for e in erros:
            st.markdown(f"- {e}")
    else:
        classe, prob_alta = classificar(modelo, renda, pessoas, pcd_idoso, moradia)
        st.subheader("Resultado da avaliação:")
        if classe == 1:
            st.error(
                f"**Alta vulnerabilidade** (probabilidade estimada: {prob_alta:.0%}). "
                "Recomenda-se prioridade no acompanhamento familiar e visita técnica do CRAS."
            )
        else:
            st.success(
                f"**Vulnerabilidade baixa/moderada** (probabilidade de alta vulnerabilidade: {prob_alta:.0%}). "
                "Família dentro dos parâmetros de acompanhamento regular."
            )
        if 0.35 <= prob_alta <= 0.65:
            st.info("Caso próximo do limite de decisão. Recomenda-se análise individual pelo técnico.")

with st.expander("Limitações da solução"):
    st.markdown(
        "- O modelo foi treinado com dados sintéticos, então o desempenho com dados reais não foi testado.\n"
        "- Usa apenas 4 variáveis. Fatores como violência doméstica, vínculos familiares, saúde e "
        "escolaridade não são considerados.\n"
        "- Nos testes, cerca de 1 em cada 5 famílias de alta vulnerabilidade não foi identificada "
        "(recall de 0,81). Um resultado verde não garante que a família não precise de apoio.\n"
        "- Rendas acima de R$ 1.500 ou mais de 10 moradores ficam fora do que o modelo viu no treino.\n"
        "- Serve para apoiar a priorização de atendimentos, não para conceder ou negar benefícios."
    )

st.markdown("---")
st.markdown(
    "<small>Trabalho Prático - Tópicos Avançados em Inteligência Artificial | "
    "Instituto Federal de Educação, Ciência e Tecnologia - Campus Catanduva</small>",
    unsafe_allow_html=True,
)
