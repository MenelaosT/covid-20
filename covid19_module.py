import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq

def preprocess_frame(df):
    df = df.groupby(by='Country/Region', as_index=False).agg('sum')
    df = df.drop(['Lat', 'Long'], 1)
    df = df.T
    df.rename(columns=df.loc["Country/Region"], inplace=True)
    df.drop(["Country/Region"], inplace=True)
    df["notChina"] = df.drop(['China'], axis=1).sum(axis=1)
    df['Day'] = np.linspace(0, df.shape[0]-1, df.shape[0], dtype = int)
    df.reset_index(inplace=True)
    df = df.rename(columns={"index":"Date"})
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    return df

def shift_to_day_zero(df, df_reference):
    for key in df.columns:
        if key!= 'Date' and key!='Day' :
            if df_reference[key].sum()>0:
                df[key] = df[key].shift(-df_reference['Day'][df_reference[key]>0].iloc[0])

def plot_confirmed_cases(df, countries):
    plt.plot(df['Day'], df['Greece'], label='Greece')
    #sns.lineplot(df['Day'], df['notChina'], label='notChina')
    for country in countries:
        plt.plot(df['Day'], df[country], label=country)
    plt.xlabel('Days since first confirmed case')
    plt.ylabel('Confirmed cases')
    plt.yscale("log")
    plt.legend()

def plot_case_death_recovery(country, df_cases, df_deaths, df_recoveries):
    plt.plot(df_cases['Day'], df_cases[country], label=country+' cases')
    plt.plot(df_recoveries['Day'], df_recoveries[country], label=country+' recovered')
    plt.plot(df_deaths['Day'], df_deaths[country], label=country+' deaths')
    plt.ylabel('entries')
    plt.yscale("log")
    plt.legend()

def func(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_cases_data(country, df):
    firstday = 0
    lastday = df[country].dropna().shape[0]

    xdata = df['Day'][(df['Day']>=firstday) & (df['Day']<lastday)]
    ydata = df[country][(df['Day']>=firstday) & (df['Day']<lastday)]

    plt.plot(xdata, ydata, 'bo', label='data')

    popt, pcov = curve_fit(func, xdata, ydata, [0.1,0.1,0.1], bounds=[[-100, -100, 0],[100, 100, 100]])
    print(popt)
    print("covariance matrix")
    print(pcov)
    x = np.linspace(firstday, lastday+5 , 100)
    plt.plot(x, func(x, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

    perr=np.sqrt(np.diag(pcov)) #standard errors
    plt.plot(x,func(x, *popt+perr), 'g--')
    plt.plot(x,func(x, *popt-perr), 'g--')

    plt.xlabel('days since first case')
    plt.ylabel('number of confirmed cases')
    plt.legend()
    #plt.yscale('log')
    plt.show()
