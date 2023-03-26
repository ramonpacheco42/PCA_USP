#%%
# Importando as Biblíotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import bartlett
from scipy.stats import chi2
from factor_analyzer import FactorAnalyzer
# %%
# Importando os dados
NotasFatorial = pd.read_excel('notas_fatorial.xlsx')
NotasFatorial
# %%
# Gerando análise descritivas para cada variável
NotasFatorial.describe()
# %%
# Scatter e ajuste linear entre as variáveis 'custos' e 'finanças'
sns.regplot(x="finanças", y="custos", data=NotasFatorial,
            color="darkorchid", scatter_kws={"s": 50}, line_kws={"lw": 1.3})

# Define os rótulos dos eixos
plt.xlabel("Finanças")
plt.ylabel("Custos")

# Mostra o gráfico
plt.show()
# %%
# Scatter e ajuste linear entre as variáveis 'custos' e 'marketing'
sns.regplot(x="marketing", y="custos", data=NotasFatorial,
            color="darkorchid", scatter_kws={"s": 50}, line_kws={"lw": 1.3})

# Define os rótulos dos eixos
plt.xlabel("marketing")
plt.ylabel("Custos")

# Mostra o gráfico
plt.show()
# %%
# Scatter e ajuste linear entre as variáveis 'custos' e 'atuária'
sns.regplot(x="atuária", y="custos", data=NotasFatorial,
            color="darkorchid", scatter_kws={"s": 50}, line_kws={"lw": 1.3})

# Define os rótulos dos eixos
plt.xlabel("atuária")
plt.ylabel("Custos")

# Mostra o gráfico
plt.show()
# %%
# Remove a coluna "estudante"
df = NotasFatorial.drop('estudante', axis=1)
# Calcula a correlação e a significância
correlations = {}
p_values = {}
for col1 in df.columns:
    for col2 in df.columns:
        if col1 != col2 and col2 != 'estudante':
            corr, p_value = pearsonr(df[col1], df[col2])
            correlations[(col1, col2)] = corr
            p_values[(col1, col2)] = p_value
# Cria o dataframe de correlação
corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['correlation'])
corr_df.index = pd.MultiIndex.from_tuples(corr_df.index)
corr_df = corr_df.unstack()

# Cria o dataframe de significância
p_df = pd.DataFrame.from_dict(p_values, orient='index', columns=['p_value'])
p_df.index = pd.MultiIndex.from_tuples(p_df.index)
p_df = p_df.unstack()
# Exibindo os dados de correlação
corr_df
# %%
# Exibindo os dados de significancia
p_df
# %%
# Criando gráfico de calor para a correlação
sns.heatmap(corr_df, annot=True, cmap='coolwarm')
# %%
# Criando gr[afico de calor para significancia
sns.heatmap(p_df, annot=True, cmap='coolwarm')
# %%
# Visualização das distribuições das variáveis, scatters, valores das correlações
sns.pairplot(NotasFatorial.iloc[:, 1:5], diag_kind='hist', plot_kws={'marker': '+'})
# %%
### Elaboração a Análise Fatorial Por Componentes Principais ###

# Selecionando apenas as colunas de índice 1 até 4:
notasfatorial = NotasFatorial.iloc[:, 1:5]

# Executando o teste de Bartlett
k = notasfatorial.shape[1]
n = notasfatorial.shape[0]
statistic, p_value = bartlett(*[notasfatorial.iloc[:, i] for i in range(k)])
df = k - 1
print(statistic)
print(p_value)
print(df)

# Obtendo o valor crítico da distribuição qui-quadrado
critical_value = chi2.ppf(q=0.95, df=df)
print(critical_value)

# Verificando se a estatística do teste é maior que o valor crítico
if statistic > critical_value:
    print("Rejeitar a hipótese nula de que as variâncias são iguais.")
else:
    print("Falha em rejeitar a hipótese nula de que as variâncias são iguais.")
#%%

# Teste de esfericidade de Bartlett
bartlett(*NotasFatorial.iloc[:, 1:5].T.values)
# %%
# Elaboração da análise fatorial por componentes principais
fatorial = FactorAnalyzer(n_factors=NotasFatorial.iloc[:, 1:5].shape[1], 
                           rotation=None, 
                           method='principal').fit(NotasFatorial.iloc[:, 1:5])
fatorial.transform(NotasFatorial.iloc[:, 1:5])
# Transforma o resultado em um DataFrame
fatorial_df = pd.DataFrame(fatorial.transform(NotasFatorial.iloc[:, 1:5]), columns=['fator_1', 'fator_2', 'fator_3', 'fator_4'])

# Exibe o DataFrame
print(fatorial_df)
# %%
# Obter os autovalores gerados pela análise fatorial
autovalores, _ = fatorial.get_eigenvalues()

# Exibe os autovalores
print(autovalores)
# %%
# Soma dos eigenvalues = 4 (quantidade de variáveis na análise)
# Também representa a quantidade máxima de possíveis fatores na análise
round(sum(autovalores),2)
# %%
# Obter a variância explicada por cada fator
variancias_fatores = fatorial.get_factor_variance()

# Transformar os resultados em um DataFrame
variancias_fatores_df = pd.DataFrame(variancias_fatores, index=['Autovalores', 'Prop. da Variância', 'Prop. da Variância Acumulada'], 
                                     columns=['Fator {}'.format(i+1) for i in range(fatorial.n_factors)])

# Exibir o DataFrame
print(variancias_fatores_df)
# %%
# Cálculo dos scores fatoriais
variancia_compartilhada_df = pd.DataFrame(fatorial.weights_)

# Atribuição de nomes às colunas
variancia_compartilhada_df = variancia_compartilhada_df.rename(columns={0: "PC1", 1: "PC2", 2: "PC3", 3: "PC4"})

# Atribuição de nomes às linhas
variancia_compartilhada_df.index = ["Finanças", "Custos", "Marketing", "Atuária"]

# Impressão da tabela formatada
print(variancia_compartilhada_df.to_markdown())
# %%
# Obter os scores fatoriais
scores_fatoriais = fatorial.transform(NotasFatorial.iloc[:, 1:5])

# Transformar os resultados em um DataFrame
scores_fatoriais_df = pd.DataFrame(scores_fatoriais, columns=['Fator {}'.format(i+1) for i in range(fatorial.n_factors)])

# Exibir o DataFrame
print(scores_fatoriais_df)
# %%
# Cálculo da correlação de Pearson entre os fatores
correlacao_fatores = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        correlacao_fatores[i, j], _ = pearsonr(scores_fatoriais_df.iloc[:, i], scores_fatoriais_df.iloc[:, j])

# Cria um DataFrame com os resultados
correlacao_fatores = pd.DataFrame(correlacao_fatores, columns=['PC1', 'PC2', 'PC3', 'PC4'], index=['PC1', 'PC2', 'PC3', 'PC4'])

# Impressão da tabela formatada
print(correlacao_fatores.to_markdown())
# %%
# Cálculo das cargas fatoriais
cargas_fatoriais = pd.DataFrame(fatorial.loadings_)

# Renomear as colunas
cargas_fatoriais.columns = ["PC1", "PC2", "PC3", "PC4"]

# Renomear as linhas
cargas_fatoriais.index = ["Finanças", "Custos", "Marketing", "Atuária"]

# Arredondar os valores para 3 casas decimais
cargas_fatoriais = cargas_fatoriais.round(3)

# Transformar em dataframe e exibir
cargas_fatoriais_df = pd.DataFrame(cargas_fatoriais)

print(cargas_fatoriais_df)
# %%
# Cálculo das comunalidades
comunalidades = fatorial.get_communalities()

# Transformar em dataframe
comunalidades_df = pd.DataFrame(comunalidades, columns=["Comunalidades"])

# Renomear as linhas
comunalidades_df.index = ["Finanças", "Custos", "Marketing", "Atuária"]

# Arredondar os valores para 3 casas decimais
comunalidades_df = comunalidades_df.round(3)

# Exibir o resultado
print(comunalidades_df)
# %%
# definindo o número de fatores
k = sum(autovalores > 1)
k
# criando o objeto fatorial2
fatorial2 = FactorAnalyzer(n_factors=k, rotation=None, method='principal')
fatorial2.fit(NotasFatorial.iloc[:, 1:5])

# calculando as comunalidades
comunalidades = fatorial2.get_communalities()

# criando um dataframe com as comunalidades
df_comunalidades = pd.DataFrame(comunalidades.round(3), index=NotasFatorial.columns[1:5], columns=['Comunalidade'])
df_comunalidades
# %%
# Criando o gráfico
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(cargas_fatoriais['PC1'], cargas_fatoriais['PC2'])

# Adicionando as setas
for i, txt in enumerate(cargas_fatoriais.index):
    ax.annotate(txt, (cargas_fatoriais['PC1'][i], cargas_fatoriais['PC2'][i]))
    ax.arrow(0, 0, cargas_fatoriais['PC1'][i], cargas_fatoriais['PC2'][i], head_width=0.05, head_length=0.05, fc='darkorchid', ec='darkorchid')

# Definindo limites do gráfico
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])

# Adicionando linhas horizontais e verticais
ax.axhline(y=0, color='orange', linestyle='--', linewidth=0.5)
ax.axvline(x=0, color='orange', linestyle='--', linewidth=0.5)

# Adicionando título e rótulos dos eixos
ax.set_title('Loading Plot', fontsize=14)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)

# Mostrando o gráfico
plt.show()
# %%
cargas_fatoriais['PC1']
# %%
# Adicionando as variaveis fator1 e fator2 no banco de dados principal
NotasFatorial = pd.concat([NotasFatorial, pd.DataFrame({"fator 1": fatorial_df.fator_1, "fator 2": fatorial_df.fator_2})], axis=1)
NotasFatorial
# %%
# Criando a caluna ranking com o calculo
NotasFatorial['ranking'] = (fatorial_df.fator_1 * variancia_compartilhada_df['PC1'][1]) + (fatorial_df.fator_2 * variancia_compartilhada_df['PC2'][1])
NotasFatorial
#%%
# Criando um dataframe em ordem descrecendo pela variavel ranking
df_sorted = NotasFatorial.sort_values(by='ranking', ascending=False)
df_sorted
# Fim!