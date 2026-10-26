CLASSIFICAÇÃO DE VULNERABILIDADE SOCIAL - VIGILÂNCIA SOCIOASSISTENCIAL

Trabalho prático da disciplina Tópicos Avançados em Inteligência Artificial, do curso de Análise e Desenvolvimento de Sistemas do IFSP Campus Catanduva, com o Prof. Dr. Márcio Andrey Teixeira, 2026.

Integrantes: Kaique Fernando, Otavio Carvalho e Rhayssa Ramos.

Link da aplicação: COLOCAR O LINK AQUI
Repositório no GitHub: COLOCAR O LINK AQUI


SOBRE O PROJETO

A aplicação ajuda técnicos do CRAS na triagem de famílias cadastradas. O técnico informa a renda per capita, a quantidade de pessoas no domicílio, se há idoso ou pessoa com deficiência e se a moradia é precária. Com esses dados o modelo classifica a família como alta vulnerabilidade, que indica prioridade de acompanhamento, ou vulnerabilidade baixa/moderada, que indica acompanhamento regular. A aplicação também mostra a probabilidade estimada de alta vulnerabilidade.

Como os microdados identificados do CadÚnico não são públicos, o modelo foi treinado com um dataset sintético gerado pelo grupo no arquivo gerar_dataset.py. Por isso os resultados servem para mostrar o funcionamento do método e não devem ser usados em casos reais.


ARQUIVOS

O arquivo gerar_dataset.py gera a base sintética, e a semente fixa faz com que o arquivo gerado seja sempre o mesmo. A base fica em dados/dados_vulnerabilidade_sintetico.csv, com 2.040 registros que incluem alguns valores ausentes e cadastros duplicados.

O treinamento.py faz a análise inicial, a limpeza, a divisão em treino e teste, a comparação entre os modelos e a avaliação. O modelo escolhido é salvo em modelo/modelo_vulnerabilidade.pkl, junto com o pré-processamento (imputação e padronização). As métricas, as matrizes de confusão e o desenho da árvore de decisão ficam na pasta resultados.

O inferencia.py tem as funções de validação das entradas e de previsão, usadas tanto pelo app.py, que é a interface em Streamlit, quanto pelo testes_casos.py, que roda os casos de teste do relatório. O requirements.txt lista as bibliotecas e versões usadas.


COMO RODAR NO COMPUTADOR

É preciso ter o Python 3.11 ou mais novo. Na pasta do projeto, instale as dependências com "pip install -r requirements.txt". O comando "python treinamento.py" treina e avalia os modelos. O comando "python testes_casos.py" roda os casos de teste. Para abrir a aplicação no navegador, use "streamlit run app.py". Rodar o gerar_dataset.py é opcional, porque o CSV já está na pasta dados.


COMO PUBLICAR NO STREAMLIT COMMUNITY CLOUD

Primeiro, crie um repositório público no GitHub e envie todos os arquivos desta pasta, incluindo as pastas modelo e dados. Depois entre em share.streamlit.io com a conta do GitHub e crie um novo app, escolhendo o repositório, a branch main e o arquivo app.py. Em Advanced settings, selecione o Python 3.12 ou 3.13 e faça o deploy. O link gerado deve ser colocado neste README e no relatório. Antes de entregar, teste o link em outro navegador.

Se o modelo for treinado de novo em outro ambiente, como o Google Colab, a versão do scikit-learn no requirements.txt precisa ser a mesma usada no treino, senão o arquivo .pkl pode não carregar.


RESULTADOS

Foram comparadas uma Regressão Logística e uma Árvore de Decisão com profundidade máxima 4. A escolha foi feita por validação cruzada de 5 folds no conjunto de treino, usando o F1 da classe de alta vulnerabilidade, e a Regressão Logística ficou um pouco à frente (0,856 contra 0,841). No conjunto de teste, com 400 famílias, ela teve precisão de 0,903, recall de 0,807 e F1 de 0,852 na classe de alta vulnerabilidade, além de acurácia de 0,887. A Árvore teve o mesmo recall, mas precisão menor (0,867). Os detalhes estão em resultados/metricas.txt.


REFERÊNCIAS

PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, v. 12, p. 2825-2830, 2011. Disponível em: https://scikit-learn.org/.

STREAMLIT INC. Streamlit Documentation. Disponível em: https://docs.streamlit.io/.

Vídeo usado como base: https://www.youtube.com/watch?v=YaFvCwjk-_A


DECLARAÇÃO DE INTEGRIDADE E USO DE FERRAMENTAS

Os autores declaram que o presente trabalho contém contribuição autoral própria. Ferramentas de inteligência artificial foram utilizadas como auxílio no desenvolvimento do código (geração do dataset sintético, treinamento, inferência e interface), na revisão textual e na estruturação do relatório. Todo o código, os dados e as análises foram executados, revisados e validados pelo grupo, que também definiu o problema, as variáveis e a interpretação dos resultados.
