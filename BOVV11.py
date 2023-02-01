# Instalar biblioteca
import yfinance as yf
import pandas as pd
from pycaret.regression import *
from pandas_profiling import ProfileReport

# Importar os Dados

dados = yf.Ticker('BOVV11.SA')
BOVV11 = dados.history(start='2000-01-01', end='2023-01-28')
BOVV11_df = pd.DataFrame(BOVV11)
# Excluir colunas sem valor para essa análise

BOVV11 = BOVV11.drop(['Dividends', 'Stock Splits'], axis=1)

# Iremos criar novas colunas com os valores referentes as médias de 7 e 30 dias

BOVV11['MM7d'] = BOVV11['Close'].rolling(window=7).mean().round(2)
BOVV11['MM30d'] = BOVV11['Close'].rolling(window=30).mean().round(2)

# Pandas Profiling

profile = ProfileReport(BOVV11, title='Previsão do preço de fechamento do BOVV11', html={'style':{'full_width':True}})
profile.to_notebook_iframe()
profile.to_file(output_file='Relatório- Previsão BOVV11.html')

# Armazenar numa variável os últimos 5 dias para realizar o teste no final

BOVV11_prever = BOVV11.tail(253)

'''# Agora iremos retirar os últimos 5 do nosso dataframe, para que nosso modelo
#  trabalhe depois com dados que ele não viu.''' 

BOVV11.drop(BOVV11.tail(253).index, inplace=True)

# Agora iremos colocar nossos dados para regredir uma casa

BOVV11['Close'] = BOVV11['Close'].shift(-1)

# Retirar os arquivos nulos

BOVV11.dropna(inplace=True)

# Resetar o index

BOVV11.reset_index(drop=True, inplace=True)
BOVV11_prever.reset_index(drop=True, inplace=True)

# Setup

setup(data=BOVV11, target='Close', session_id=123, remove_perfect_collinearity = False)

# Comparar os 3 modelos com melhor performace

top3 = compare_models(n_select=3)

lar = create_model('lar', fold=10)


# Modo Tuning

params = {'alpha': [0.02, 0.024, 0.025, 0.026, 0.03]}
tune_lar = tune_model(lar, n_iter=100, optimize = 'RMSE')
# Plotar o nosso modelo 

plot_model(lar, plot='error')

plot_model(tune_lar, plot='feature')


# Predict model

predict_model(tune_lar)

#finalizando o modelo

final_tune_lar = finalize_model(tune_lar)

# Agora testar o modelo no dataframe que salvamos com as últimas 15 linhas

prev = predict_model(final_tune_lar, data=BOVV11_prever)

fig = px.line(round(prev[['Close','Label']],2),
                x = round(prev[['Close', 'Label']],2).index,
                y = ['Close', 'Label'],
                title = 'Preço fechamento x preço previsto de BOVV11 ',
                width = 1500, height = 1000)
fig.show()
                
                
                
                
            













