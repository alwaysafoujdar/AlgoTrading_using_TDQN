import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot as plt


class TimeSeriesAnalyser:

    def __init__(self, timeSeries):
        self.timeSeries = timeSeries


    def plotTimeSeries(self):
        pd.plotting.register_matplotlib_converters()
        plt.figure(figsize=(10, 4))
        plt.plot(self.timeSeries.index, self.timeSeries.values, color='blue')
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.show()

    
    def timeSeriesDecomposition(self, model='multiplicative'):
        # Decomposition of the time series and plotting of each component
        # period=5 because there are 5 trading days in a week, and the decomposition looks for weekly seasonality
        # period=21 should be used for monthly seasonality and period=252 for yearly seasonality
        decomposition = seasonal_decompose(self.timeSeries, model=model, period=5, extrapolate_trend='freq')
        plt.rcParams.update({'figure.figsize': (16,9)})
        decomposition.plot()
        plt.show()


    def stationarityAnalysis(self):
        """
        GOAL: Assess whether or not the time series is stationary.
        
        INPUTS: /
        
        OUTPUTS: /
        """

        # Augmented Dickey-Fuller test
        print("Stationarity analysis: Augmented Dickey-Fuller test (ADF):")
        results = adfuller(self.timeSeries, autolag='AIC')
        print("ADF statistic: " + str(results[0]))
        print("p-value: " + str(results[1]))
        print('Critial values (the time series is not stationary with X% condifidence):')
        for key, value in results[4].items():
            print(str(key) + ': ' + str(value))
        if results[1] < 0.05:
            print("The ADF test affirms that the time series is stationary.")
        else:
            print("The ADF test could not affirm whether or not the time series is stationary...")


    def cyclicityAnalysis(self):
        """
        GOAL: Assess whether or not the time series presents a significant
              seasonality component.
        
        INPUTS: /
        
        OUTPUTS: /
        """

        # Generation of an Autoacorrelation function plot
        plt.rcParams.update({'figure.figsize': (16,9)})
        pd.plotting.autocorrelation_plot(self.timeSeries)
        plt.show()

        # Generation of both the autocorrelation and the partial autocorrelation plots
        _, axes = plt.subplots(2, figsize=(16, 9))
        plot_acf(self.timeSeries, lags=21, ax=axes[0])
        plot_pacf(self.timeSeries, lags=21, ax=axes[1])
        plt.show()

        # Generation of several lag plots
        _, axes = plt.subplots(1, 10, figsize=(17, 9), sharex=True, sharey=True)
        for i, ax in enumerate(axes.flatten()[:10]):
            pd.plotting.lag_plot(self.timeSeries, lag=i+1, ax=ax)
            ax.set_title('Lag ' + str(i+1))
        plt.show()
        