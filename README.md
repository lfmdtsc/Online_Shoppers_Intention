# Online_Shoppers_Intention

### Agrupamento hierárquico
### Neste projeto vamos usar a base online shoppers purchase intention de Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018). Web Link.

### A base trata de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de 12 meses, para posteriormente estudarmos a relação entre o design da página e o perfil do cliente - "Será que clientes com comportamento de navegação diferentes possuem propensão a compra diferente?"

### Nosso objetivo agora é agrupar as sessões de acesso ao portal considerando o comportamento de acesso e informações da data, como a proximidade a uma data especial, fim de semana e o mês.

# LINK DA FERRAMENTA:
https://online-shoppers-intention.onrender.com

# CÓDIGO DA FERRAMENTA:

# Imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff

from PIL                 import Image
from io                  import BytesIO

@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Função para converter o df para excel
@st.cache
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


# Função principal da aplicação

def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'Online Shoppers Intentions', \
        page_icon = './img/OSI_02.jpeg',
        layout="wide",
        initial_sidebar_state='expanded'
    )
    # Título principal da aplicação
    st.write('# Online Shoppers Intentions - Análise por Agrupamento')
    st.markdown("---")
    
    # Apresenta a imagem na barra lateral da aplicação
    image = Image.open("./img/OSI_01.jpeg")
    st.sidebar.image(image)

    st.write('##### Agrupamento hierárquico')

    st.write('###### Neste projeto vamos usar a base online shoppers purchase intention de Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Neural Comput & Applic (2018). Web Link. A base trata de registros de 12.330 sessões de acesso a páginas, cada sessão sendo de um único usuário em um período de 12 meses, para posteriormente estudarmos a relação entre o design da página e o perfil do cliente - "Será que clientes com comportamento de navegação diferentes possuem propensão a compra diferente?" Nosso objetivo agora é agrupar as sessões de acesso ao portal considerando o comportamento de acesso e informações da data, como a proximidade a uma data especial, fim de semana e o mês.')

    st.write('|Variavel                |Descrição          |\n'
             '|------------------------|:-------------------|\n'
             '|Administrative          | Quantidade de acessos em páginas administrativas|\n' 
             '|Administrative_Duration | Tempo de acesso em páginas administrativas | \n'
             '|Informational           | Quantidade de acessos em páginas informativas  | \n'
             '|Informational_Duration  | Tempo de acesso em páginas informativas  | \n'
             '|ProductRelated          | Quantidade de acessos em páginas de produtos | \n'
             '|ProductRelated_Duration | Tempo de acesso em páginas de produtos | \n'
             '|BounceRates             | *Percentual de visitantes que entram no site e saem sem acionar outros *requests* durante a sessão  | \n'
             '|ExitRates               | * Soma de vezes que a página é visualizada por último em uma sessão dividido pelo total de visualizações | \n'
             '|PageValues              | * Representa o valor médio de uma página da Web que um usuário visitou antes de concluir uma transação de comércio eletrônico | \n'
             '|SpecialDay              | Indica a proximidade a uma data festiva (dia das mães etc) | \n'
             '|Month                   | Mês  | \n'
             '|OperatingSystems        | Sistema operacional do visitante | \n'
             '|Browser                 | Browser do visitante | \n'
             '|Region                  | Região |\n '
             '|TrafficType             | Tipo de tráfego                  | \n'
             '|VisitorType             | Tipo de visitante: novo ou recorrente | \n'
             '|Weekend                 | Indica final de semana | \n'
             '|Revenue                 | Indica se houve compra ou não |\n'
              '* variávels calculadas pelo google analytics.')
    st.markdown("---")
    
    # Botão para carregar arquivo na aplicação
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Online_Shoppers_Intention", type = ['csv','xlsx'])

    # Verifica se há conteúdo carregado na aplicação
    if (data_file_1 is not None):
        df = pd.read_csv(data_file_1)
        
        st.write(df.head(100))
        
        sidebar = st.sidebar
        
        mode = sidebar.radio("Mode", ["K-Means", "Aglomerativo/Hierarquico"])
        # st.markdown("<h1 style='text-align: center; color: #ff0000;'>COVID-19</h1>", unsafe_allow_html=True)
        st.markdown("#### Mode: {}".format(mode), unsafe_allow_html=True)
        
        if mode=="Aglomerativo/Hierarquico":    
        
            df.Revenue.value_counts(dropna=False)
        
            st.write('## Análise descritiva')
            st.write('### Verificado a distribuição dessas variáveis')
        
            st.write(df.describe())
        
            st.write('### Verificado se existe valores *missing*')

            st.write(df.isnull().sum())

            st.write('### Verificando valores únicos de cada variável')

            st.write(df.nunique(axis=0))
        
            st.markdown("---")
        
            st.write('#### Variáveis Descartadas Nessa Análise:')
        
                
            df_clean = df.drop(['Browser','OperatingSystems','Region','TrafficType','VisitorType', 'ExitRates', 'PageValues'], axis=1)
            st.write('###### Browser - OperatingSystems - Region - TrafficType - VisitorType - ExitRates - PageValues')
        
            st.markdown("---")
        
            st.write('#### Transfromando as variaveis qualitativas em dummies:')
        
            df_2 = pd.get_dummies(df_clean.dropna())
            st.write(df_2.head())
        
            st.write('#### Informando as variáveis categóricas que serão utilizadas no algoritmo:')
        
            vars_cat = [True if x in {'Weekend', 'Revenue', 'Month_Aug','Month_Dec','Month_Feb','Month_Jul','Month_June','Month_Mar','Month_May','Month_Nov','Month_Oct','Month_Sep'} else False for x in df_2.columns]
            st.write(vars_cat)
        
            df_2.shape
        
            st.markdown("---")

            st.write('### Número de grupos')
            st.write('###### Neste projeto vamos adotar uma abordagem bem pragmática e avaliar agrupamentos hierárquicos com 3 e 4 grupos, por estarem bem alinhados com uma expectativa e estratégia do diretor da empresa.')
        
            from gower import gower_matrix
            from scipy.spatial.distance import pdist, squareform
            from scipy.cluster.hierarchy import linkage, fcluster  # Adicionando os imports do scipy
        
            distancia_gower = gower_matrix(df_2, cat_features=vars_cat)
        
            gdv = squareform(distancia_gower,force='tovector')
        
            Z = linkage(gdv, method='complete')

            # Selecionar o Número de Grupos:
            n_grupos = sidebar.selectbox("Selecionando o Número de Grupos:", ["03","04"])
            
            if n_grupos=="03":
            
                st.write('### 3 Grupos:')
        
                df_2['grupo_3'] = fcluster(Z, 3, criterion='maxclust')
                df_2['grupo_4'] = fcluster(Z, 4, criterion='maxclust')
                st.write(df_2.grupo_3.value_counts())
        
                st.write('#### Unificando os dataframes')
                
                st.write(df_2.grupo_3.value_counts())
                
                df_date = df.reset_index().merge(df_2.reset_index(), how='left')
                
                st.write('### Análise Descritiva das Compras Efetivadas pelos Grupos Durante a Semana e nos Finais de Semana:')
                
                st.write(df_date.groupby(['Weekend','Revenue', 'grupo_3'])['index'].count().unstack().fillna(0).style.format(precision=0))
                
                st.write('### Análise Descritiva das Compras Efetivadas pelos Grupos nos Meses do Ano:')
                
                st.write(df_date.groupby([ 'Month','Revenue', 'grupo_3'])['index'].count().unstack().fillna(0).style.format(precision=0))
                
                st.write(df_date.groupby([ 'SpecialDay','Month', 'grupo_3'])['index'].count().unstack().fillna(0).style.format(precision=0))
                
                st.write('### Análise Descritiva com Relação a Navegação no Site pelos Grupos:')
                
                st.write(df_date.groupby([ 'ProductRelated', 'Revenue', 'grupo_3'])['index'].count().unstack().fillna(0).style.format(precision=0))

                st.write(df_date.groupby([ 'Administrative', 'Revenue',  'grupo_3'])['index'].count().unstack().fillna(0).style.format(precision=0))

                st.write(df_date.groupby(['Revenue', 'grupo_3'])['index'].count().unstack().fillna(0).style.format(precision=0))
                
                
            if n_grupos=="04":
                st.write('### 4 Grupos:')
        
                df_2['grupo_4'] = fcluster(Z, 4, criterion='maxclust')
                st.write(df_2.grupo_4.value_counts())
        
                st.markdown("---")
        
                st.write('#### Unificando os dataframes')
        
                df_date = df.reset_index().merge(df_2.reset_index(), how='left')
        
                st.write(df_2.grupo_4.value_counts())
        
                st.write('### Análise Descritiva das Compras Efetivadas pelos Grupos Durante a Semana e nos Finais de Semana:')
        
                st.write(df_date.groupby([ 'Weekend','Revenue', 'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0))
        
                st.write('### Análise Descritiva das Compras Efetivadas pelos Grupos nos Meses do Ano:')
        
                st.write(df_date.groupby([ 'Month','Revenue', 'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0))
                
                st.write(df_date.groupby([ 'SpecialDay','Month', 'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0))
                
                st.write('### Análise Descritiva com Relação a Navegação no Site pelos Grupos:')
                
                st.write(df_date.groupby([ 'ProductRelated', 'Revenue', 'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0))
                
                st.write(df_date.groupby([ 'Administrative', 'Revenue',  'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0))
        
                st.write(df_date.groupby(['Revenue', 'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0))
                        
            st.markdown("---")
        
            st.write('###### Com relação a escolha da quantidade de grupos, entre 3 e 4, no meu ponto de vista, a escolha pelo número de 4 grupos será mais efetiva. Pois, quando se adiciona 1 grupo a mais, além dos 3 pretendidos, percebe-se que o grupo 3 é subdividido em 2, ficando o grupo 3 condensado apenas nas pessoas que utilizaram os sites no mês de maio, com o restante sendo realocado no grupo 4. E nesse grupo, apenas em 3 ocasiões (em fevereiro) as compras foram efetivadas com sucesso.')
            st.write('###### A relação da quantidade de acesso em páginas administrativas com a efetuação de compra não possui muita relação. Porém, poucas pessoas com poucos acessos a páginas de produtos finalizam a compra de forma efetiva. No geral, é necessários mais acessos as páginas de produtos para que seja finalizada a compra com sucesso.')
        
            st.write('## Avaliação de resultados:')
        
            st.write(df_date.groupby(['Revenue', 'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0))

            
            df_sns = df_date.groupby(['Revenue', 'grupo_4'])['index'].count().unstack().fillna(0)
            
            #plots
            #plots
            fig, ax = plt.subplots(4,1,figsize=(15,20), constrained_layout=True)
            plt.rcParams['legend.fontsize'] = 10
            plt.rcParams["legend.title_fontsize"] = 12
            sns.barplot(data=df_sns, x=['Não Comprou', 'Comprou'], y=df_sns[1], ax=ax[0])
            ax[0].tick_params(axis='x', rotation=45, labelsize=10)
            ax[0].tick_params(axis='y', labelsize=10)
            ax[0].set_ylabel( "Número de Clientes" , size = 12 )
            ax[0].set_xlabel( "Grupo 01" , size = 12 )
            sns.barplot(data=df_sns, x=['Não Comprou', 'Comprou'], y=df_sns[2], ax=ax[1])
            ax[1].tick_params(axis='x', rotation=45, labelsize=10)
            ax[1].tick_params(axis='y', labelsize=10)
            ax[1].set_ylabel( "Número de Clientes" , size = 12 )
            ax[1].set_xlabel( "Grupo 02" , size = 12 )
            sns.barplot(data=df_sns, x=['Não Comprou', 'Comprou'], y=df_sns[3], ax=ax[2])
            ax[2].tick_params(axis='x', rotation=45, labelsize=10)
            ax[2].tick_params(axis='y', labelsize=10)
            ax[2].set_ylabel( "Número de Clientes" , size = 12 )
            ax[2].set_xlabel( "Grupo 03" , size = 12 )
            sns.barplot(data=df_sns, x=['Não Comprou', 'Comprou'], y=df_sns[4], ax=ax[3])
            ax[3].tick_params(axis='x', rotation=45, labelsize=10)
            ax[3].tick_params(axis='y', labelsize=10)
            ax[3].set_ylabel( "Número de Clientes" , size = 12 )
            ax[3].set_xlabel( "Grupo 04" , size = 12 )
            sns.despine()
            st.pyplot(plt)        
                                  
            # # Group data together
            # hist_data = df_date.groupby(['Revenue', 'grupo_4'])['index'].count().unstack().fillna(0).style.format(precision=0)

            # group_labels = df_date['Revenue']

            # # Create distplot with custom bin_size
            # fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])
               
            # # Plot!
            # st.plotly_chart(fig, use_container_width=True)
            
            st.write('#### De acordo com as análises, o grupo 2 é mas propenso a efetuar a compra do que os outros.')
        
        if mode=="K-Means": 
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            
            df_clean_kmeans = df.drop(['Month','Browser','OperatingSystems','Region','TrafficType','Weekend', 'VisitorType'], axis=1)
        
            st.markdown("---")
        
            st.write('#### Transformando as variaveis qualitativas em dummies:')
        
            df_2 = pd.get_dummies(df_clean_kmeans.dropna())
            st.write(df_2.head())
        
            st.write('#### Padronizando as Variaveis Quantitativas:')
            
            df_pad = pd.DataFrame(StandardScaler().fit_transform(df_clean_kmeans), columns =df_clean_kmeans.columns)
            st.write(df_pad.head())
            
            df_pad_escopo = df_pad[['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration']]

       
            st.write('#### Número de grupos')
            st.write('###### Neste projeto vamos adotar uma abordagem bem pragmática e avaliar agrupamentos hierárquicos com 3 e 4 grupos, por estarem bem alinhados com uma expectativa e estratégia do diretor da empresa.')
       
            # Selecionar o Número de Grupos:
            num_clusters = sidebar.selectbox("Selecionando o Número de Clusters:", ["03","04"])
            
            if num_clusters=="03":
        
                st.write('#### 3 Clusters:')
        
                cluster = KMeans(n_clusters=3, random_state=0)
                cluster.fit(df_pad_escopo)

                st.write('##### Agrupando os Grupos:')
                cluster.labels_

                df_pad['grupos'] = pd.Categorical(cluster.labels_)
                df_pad_escopo['grupos'] = pd.Categorical(cluster.labels_)
        
        
                st.write('### Visualizando a distribuição pelo seaborn - pairplot:')

                plot_3 = sns.pairplot(df_pad_escopo, hue='grupos')
                
                st.pyplot(plot_3.fig)
            
            if num_clusters=="04":
            
                st.write('### 4 Clusters:')
        
                cluster = KMeans(n_clusters=4, random_state=0)
                cluster.fit(df_pad_escopo)

                cluster.labels_

                df_pad['grupos'] = pd.Categorical(cluster.labels_)
                df_pad_escopo['grupos'] = pd.Categorical(cluster.labels_)
        
        
                st.write('### Visualizando a distribuição pelo seaborn - pairplot:')

                plot_4 = sns.pairplot(df_pad_escopo, hue='grupos')
                
                st.pyplot(plot_4.fig)
                

if __name__ == '__main__':
	main()
