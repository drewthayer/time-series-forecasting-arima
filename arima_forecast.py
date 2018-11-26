import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    data = pd.read_csv('data/IPG2211A2N.csv', header=0)
    # columns = ['DATE', 'IPG2211A2N'], shape (958, 1)
    data.index = pd.to_datetime(data['DATE'])
    data.drop(['DATE'], axis=1, inplace=True)
    data.columns = ['energy']

    # quick plot
    #import plotly.plotly as ply
    #import cufflinks as cf
    #data.iplot(title="Energy Production Jan 1985--Jan 2018")
    write_plot = False
    if write_plot:
        fig = plt.subplots(1,1,figsize=(8,4))
        plt.plot(data.energy)
        plt.title("Energy Production Jan 1985--Jan 2018")
        plt.savefig('figures/fig1.png', dpi=250)
        plt.close()

    # decompose timeseries
    from statsmodels.tsa.seasonal import seasonal_decompose
    result = seasonal_decompose(data, model='multiplicative') # A object with seasonal, trend, and resid attributes

    plot_decompose = False
    if plot_decompose:
        fig, ax = plt.subplots(4,1)
        ax[0].plot(data, '-k', label='observed')
        ax[0].set_ylabel('observed')
        ax[0].get_xaxis().set_ticklabels([])
        ax[1].plot(result.trend, '-b', label='trend')
        ax[1].set_ylabel('trend')
        ax[1].get_xaxis().set_ticklabels([])
        ax[2].plot(result.seasonal, '-g', label='seasonal')
        ax[2].set_ylabel('seasonal')
        ax[2].get_xaxis().set_ticklabels([])
        ax[3].plot(result.resid, '-k', label='residual')
        ax[3].set_ylabel('residual')
        plt.suptitle('time series decomposition')
        plt.subplots_adjust(top=0.9)
        #plt.tight_layout()
        plt.show()

    # pyramid-arima stepwise parameter grid search
    from pyramid.arima import auto_arima
    stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

    print(stepwise_model.aic())

    # train test split
    splitdate = '2015-12-01'
    train = data.loc[:splitdate]
    test = data.loc['2016-01-01':]

    # fit model to training data
    stepwise_model.fit(train)

    # predict forecast
    forecast = stepwise_model.predict(n_periods=len(test))

    plot_forecast = False
    if plot_forecast:
        plt.plot(test.values, 'k', label='true')
        plt.plot(forecast, '--b', label='forecast')
        plt.title('arima forecast')
        plt.legend()
        plt.show()
