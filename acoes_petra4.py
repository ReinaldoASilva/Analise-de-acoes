


import yfinance as yf 
from yfinance import Ticker
from pycaret.regression import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.base import _pprint
codigo = ['OIBR3.SA']
oibr3 = yf.Ticker(codigo[0])
df_oibr3 = oibr3.history(period='5y')
#removendo coluna
df_oibr3 = df_oibr3.drop(['Dividends', 'Stock Splits'], axis=1)

#criar novas colunas
# criando a coluna mm7d que irá receber do df a coluna Close
#rolling é para inserir o valor da media
# mean para criar uma média
#round e para ter somente duas casas decimais
df_oibr3['MM7d'] = df_oibr3['Close'].rolling(window=7).mean().round(2)
df_oibr3['MM30d'] = df_oibr3['Close'].rolling(window=30).mean().round(2)

# dias para previsão 
df_oibr3_previsao = df_oibr3.tail(5)

# retirar 5 dias para previsão

df_oibr3.drop(df_oibr3.tail(5).index, inplace=True)

# empurrar os valore das ações para frente 

df_oibr3['Close'] = df_oibr3['Close'].shift(-1)

#retirar os campos nulos

df_oibr3.dropna(inplace=True)

#drop index

df_oibr3.reset_index(drop=True, inplace=True)
df_oibr3_previsao.reset_index(drop=True, inplace=True)

setup(data= df_oibr3, target= 'Close', session_id= 123)
df_oibr3.columns
setup(data= df_oibr3, target='Close', session_id=123)








'''

setup(data=df_oibr3, target='Close', session_id=123)
# calcular a media movel de 200 periodos 
df_oibr3['Media_Movel'] = df_oibr3['Close'].rolling(window=3).mean() 

# plotar o grafico com a media movel de 200 periodos 
plt.figure(figsize=(16,8)) 
plt.plot(df_oibr3['Close']) 
plt.plot(df_oibr3['Media_Movel']) 
plt.xlabel('Data') 
plt.ylabel('Preço das Ações') 
plt.title('Petrobras (OIBR3)') 
plt.legend(['Preço das Ações', 'Média Móvel de 200 Períodos']) 
plt.show()



# Selecionar as colunas relevantes como recursos
X = df_oibr3[['Volume', 'High', 'Low']]

# Definir o preço de fechamento como alvo
y = df_oibr3['Close']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Treinar o modelo de regressão linear
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Fazer previsões com o conjunto de teste
y_pred = regressor.predict(X_test)

# Avaliar a precisão do modelo
accuracy = regressor.score(X_test, y_test)
print("Acurácia:", accuracy)'''