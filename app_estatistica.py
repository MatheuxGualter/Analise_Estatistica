import streamlit as st
import pandas as pd
from scipy import stats
import plotly.express as px

# --- Configuração da Página ---
st.set_page_config(
    page_title="Análise de Regressão: Preço vs. Área de Imóveis",
    page_icon="🏠",
    layout="wide"
)

# --- Função para Carregar e Preparar os Dados ---
@st.cache_data
def carregar_dados(caminho_arquivo):
    """
    Carrega os dados do CSV, renomeia colunas, remove valores nulos
    e outliers usando o método IQR.
    """
    df = pd.read_csv(caminho_arquivo)
    
    # Renomear colunas para facilitar o uso
    df.rename(columns={
        'LR_DR_CURRENTPRICE': 'Preco',
        'EPC_DR_FLOOR_AREA': 'Area'
    }, inplace=True)

    # Remover linhas onde 'Preco' ou 'Area' são nulos ou zero
    df.dropna(subset=['Preco', 'Area'], inplace=True)
    df = df[(df['Preco'] > 0) & (df['Area'] > 0)]

    # Remover outliers para uma análise mais robusta
    for col in ['Preco', 'Area']:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        limite_inferior = q1 - 1.5 * iqr
        limite_superior = q3 + 1.5 * iqr
        df = df[(df[col] >= limite_inferior) & (df[col] <= limite_superior)]
        
    return df

# --- Carregamento dos Dados ---
try:
    df_imoveis = carregar_dados('AddressSpineUK.csv')
except FileNotFoundError:
    st.error("Erro: O arquivo 'AddressSpineUK.csv' não foi encontrado. Por favor, certifique-se de que ele está na mesma pasta que o script Python.")
    st.stop()


# --- Cálculos de Regressão e Correlação ---
# Usando scipy.stats.linregress para obter todos os valores de uma vez
slope, intercept, r_value, p_value, std_err = stats.linregress(df_imoveis['Area'], df_imoveis['Preco'])
r_squared = r_value**2


# --- Interface da Aplicação ---

st.title("🏠 Análise de Regressão Linear: Preço vs. Área de Imóveis")
st.markdown("Análise da relação entre a área total e o preço de imóveis na região de Bristol, Reino Unido.")

st.markdown("---")

# --- 1. Contexto e Pergunta ---
st.header("1. Contexto e Pergunta da Análise")
st.markdown("""
A base de dados **Address Spine** do Reino Unido contém diversas características sobre propriedades residenciais. Para esta análise, focamos em dados da cidade de **Bristol**, utilizando apenas registros onde a origem da informação é confirmada como **'Actual' (Real)**, garantindo maior qualidade e confiabilidade.

A avaliação da relação entre a área de um imóvel e seu preço é fundamental para o mercado imobiliário, pois permite a criação de modelos de precificação, identificação de oportunidades e entendimento das tendências de valorização. A área é, frequentemente, o principal fator que influencia o valor de um imóvel.

Com base neste cenário, formulamos as seguintes perguntas:
> **Há uma relação linear entre a Área Total de um imóvel (em m²) e seu Preço Estimado Atual (em £)? Qual a sua intensidade? Podemos usar a Área para prever o Preço de um imóvel?**
""")

st.markdown("---")

# --- 2. Análise Exploratória e Gráfico ---
st.header("2. Análise Exploratória dos Dados")
st.write("Abaixo, uma amostra dos dados já limpos, sem valores nulos e com outliers removidos, que serão utilizados na análise:")
st.dataframe(df_imoveis.head())

st.subheader("Gráfico de Dispersão")
fig = px.scatter(
    df_imoveis,
    x='Area',
    y='Preco',
    title='Relação entre Área Total (m²) e Preço (£)',
    labels={'Area': 'Área Total (m²)', 'Preco': 'Preço Estimado Atual (£)'},
    trendline='ols', # 'ols' significa "Ordinary Least Squares", que adiciona a reta de regressão
    trendline_color_override='red'
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- 3. Correlação e Regressão ---
st.header("3. Correlação de Pearson (r) e Regressão Linear")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Correlação de Pearson (r)")
    st.metric(label="Valor de r", value=f"{r_value:.4f}")
    st.write(f"""
    O coeficiente de correlação de Pearson (r) de **{r_value:.4f}** indica uma **correlação positiva forte** entre a área e o preço do imóvel.
    - **Sentido**: Positivo, ou seja, quando a área aumenta, o preço também tende a aumentar.
    - **Força**: Forte, pois o valor está consideravelmente próximo de +1.
    """)

with col2:
    st.subheader("Coeficiente de Determinação (R²)")
    st.metric(label="Valor de R²", value=f"{r_squared:.4f}")
    st.write(f"""
    O R² é de **{r_squared:.4f}**, o que significa que aproximadamente **{r_squared:.1%}** da variabilidade no preço dos imóveis pode ser explicada pela variabilidade na sua área total. 
    Os outros {1-r_squared:.1%} são influenciados por outros fatores não incluídos no modelo (localização exata, condição do imóvel, número de quartos, etc.).
    """)

st.subheader("Equação da Reta de Regressão Linear")
st.latex(f"Preço = {intercept:,.2f} + {slope:,.2f} \\times Área")

st.markdown(f"""
A equação da reta que modela a relação é: **Preço = {intercept:,.2f} + {slope:,.2f} * Área**.

- **Coeficiente 'a' (Intercepto) = {intercept:,.2f}**: Este é o valor teórico do imóvel se sua área fosse 0 m². Neste contexto, ele não possui uma interpretação prática, servindo como um ponto de partida para o modelo matemático.

- **Coeficiente 'b' (Inclinação) = {slope:,.2f}**: Este é o coeficiente mais importante. Ele indica que, em média, para **cada metro quadrado (m²) adicional** na área de um imóvel na região de Bristol, o seu preço **aumenta em aproximadamente £ {slope:,.2f}**.
""")

st.markdown("---")

# --- 4. Previsão Pontual ---
st.header("4. Faça uma Previsão Pontual")
st.write("Use o modelo de regressão para estimar o preço de um imóvel com base na sua área.")
area_para_prever = st.number_input(
    label="Digite a área do imóvel em m²:",
    min_value=float(df_imoveis['Area'].min()),
    max_value=float(df_imoveis['Area'].max()),
    value=100.0, # Um valor padrão plausível
    step=5.0
)

if area_para_prever:
    preco_previsto = intercept + slope * area_para_prever
    st.success(f"O preço estimado para um imóvel com **{area_para_prever:.2f} m²** é de **£ {preco_previsto:,.2f}**.")

st.markdown("---")

# --- 5. Conclusão ---
st.header("5. Conclusão")
st.markdown("""
- **Síntese**: A análise confirmou uma **relação positiva e forte** entre a área total e o preço dos imóveis em Bristol, com um modelo de regressão linear que explica cerca de **44.4%** da variação dos preços.

- **Utilidade e Limitações**: O modelo é útil para se ter uma **estimativa geral** de valor, mas não deve ser usado para uma avaliação precisa, pois ignora fatores cruciais como a localização exata dentro de Bristol, o estado de conservação, o número de quartos e o acabamento do imóvel.

- **Desafios dos Dados**: Um dos principais desafios para a obtenção e tratamento destes dados foi a necessidade de uma limpeza rigorosa. Muitos registros possuíam dados faltantes ou eram "derivados" (estimados por modelos), o que exigiu a aplicação de filtros (como `SOURCE_HTYPE_HYBRID_B = 'Actual'`) para garantir a qualidade. Além disso, a remoção de outliers foi uma etapa crucial para evitar que imóveis de altíssimo luxo ou propriedades anômalas distorcessem o modelo de regressão geral.
""")