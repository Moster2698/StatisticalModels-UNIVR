from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

#Execute an AdFuller test on the data. If the p-value is less than <=0.05 the data is stationary, otherwise it is not.
def adfuller_test(data):
  adf_test = adfuller(data,autolag='AIC') # AIC is the default option
  print('ADF Statistic:', adf_test[0])
  print('p-value: ', adf_test[1])
  print('Critical Values:')
  for key, value in adf_test[4].items():
      print('\t%s: %.3f' % (key, value))
  if adf_test[1] <= 0.05:
    print('We can reject the null hypothesis (H0) --> data is stationary')
  else:
    print('We cannot reject the null hypothesis (H0) --> data is non-stationary')
#Execute a KPSS test on the data passed by the argument. If the p-value is less than 0.05 then null hypothesis cannot be rejected
# and that's means the data is not trend stationary, otherwise is trend stationary.
def kpss_test(data):
  kpss_out = kpss(data,regression='ct', nlags='auto', store=True)
  print('KPSS Statistic:', kpss_out[0])
  print('p-value: ', kpss_out[1])
  if kpss_out[1] <= 0.05:
    print('We can reject the null hypothesis (H0) --> data is not trend stationary')
  else:
    print('We cannot reject the null hypothesis (H0) --> data is trend stationary')

def plot_autocorr(data, lags):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    plot_acf(data, lags=lags, ax=ax[0])
    plot_pacf(data, lags=lags, ax=ax[1])


def spectral_analyisis(data, Fs):
  ###Retrieve the seasonlity by using the periodogram and getting the most used frequencies.
  f_per, Pxx_per = signal.periodogram(data,Fs,detrend=None,window='hann',return_onesided=True,scaling='density')
  f_per = f_per[1:]
  Pxx_per = Pxx_per[1:]

  #Find the peaks of the periodogram.
  peaks = signal.find_peaks(Pxx_per[f_per >= 0], prominence=100000)[0]
  peak_freq = f_per[peaks]
  peak_dens = Pxx_per[peaks]

  #Plot of the analysis transformation and of its peaks
  plt.plot(peak_freq[:5], peak_dens[:5], 'o');
  plt.plot(f_per[2:],Pxx_per[2:])

  #Retrieving of the values
  data = {'Frequency': peak_freq, 'Density': peak_dens, 'Period': 1/peak_freq}
  df = pd.DataFrame(data)
  print(df.head())
  plt.plot(f_per, Pxx_per)
  plt.xlabel('Sample Frequencies')
  plt.ylabel('Power')
  #We see that there are multiple frequencies that repeats. I've decided to focus on the 365 period which represents an yearly seasonality
  #and then focus into the 90 days which is a 3 month seasonality. 