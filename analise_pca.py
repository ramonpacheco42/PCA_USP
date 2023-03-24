#%%
# Importando as Biblíotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import bartlett
from scipy.stats import chi2
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

# Suponha que a matriz de dados seja armazenada na variável "notas_fatorial"
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
notasfatorial.shape[0]
# %%
