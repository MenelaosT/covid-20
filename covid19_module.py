import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
from scipy.stats import chisquare
import uncertainties as unc
import uncertainties.unumpy as unp
import seaborn as sns
sns.set_style('whitegrid')
mpl.rcParams['figure.dpi'] = 150


def preprocess_frame(df):
    df = df.groupby(by='Country/Region', as_index=False).agg('sum')
    df = df.drop(['Lat', 'Long'], 1)
    df = df.T
    df.rename(columns=df.loc["Country/Region"], inplace=True)
    df.drop(["Country/Region"], inplace=True)
    df["notChina"] = df.drop(['China'], axis=1).sum(axis=1)
    df['Day'] = np.linspace(0, df.shape[0] - 1, df.shape[0], dtype=int)
    df.reset_index(inplace=True)
    df = df.rename(columns={"index": "Date"})
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    return df


def shift_to_day_zero(df, df_reference):
    for key in df.columns:
        if key != 'Date' and key != 'Day':
            if df_reference[key].sum() > 0:
                df[key] = df[key].shift(-df_reference['Day']
                                        [df_reference[key] > 0].iloc[0])


def set_cases_labels():
    plt.xlabel('Days since first confirmed case', fontsize="x-large")
    plt.ylabel('Confirmed cases', fontsize="x-large")


def set_deaths_labels():
    plt.xlabel('Days since first death', fontsize="x-large")
    plt.ylabel('Number of deaths', fontsize="x-large")


def plot_growth(df, countries, case):
    plt.figure(figsize=(15, 7))
    for country in countries:
        plt.plot(df['Day'], df[country], label=country)
    plt.plot(df['Day'], df['Greece'], label='Greece')
    if case == "confirmed":
        set_cases_labels()
    elif case == "deaths":
        set_deaths_labels()
    plt.yscale("log")
    plt.legend(fontsize="x-large")
    plt.show()


def plot_case_death_recovery(country, df_cases, df_deaths, df_recoveries):
    plt.plot(df_cases['Day'], df_cases[country], label=country + ' cases')
    plt.plot(df_recoveries['Day'], df_recoveries[country],
             label=country + ' recovered')
    plt.plot(df_deaths['Day'], df_deaths[country], label=country + ' deaths')
    plt.ylabel('entries')
    plt.yscale("log")
    plt.legend()


def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def logistic(x, a, b, c, d, e):
    return a / (1 + b * np.exp(-c * (x - d))) + e


def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))


def R2(y, y_fit):
    res = y - y_fit
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res / ss_tot)


def fit_exponential(country, df, interval):
    firstday = interval[0]
    lastday = interval[1]
    xdata = df['Day'][(df['Day'] >= firstday) & (df['Day'] < lastday)]
    ydata = df[country][(df['Day'] >= firstday) & (df['Day'] < lastday)]
    popt, pcov = curve_fit(exponential, xdata, ydata, [0.1, 0.1, 0.1], bounds=(
        [0, 0, -1e2], [1e3, 1, 1e3]), maxfev=1e5)
    Rsq = R2(ydata, exponential(xdata, *popt))
    ae, be, ce = unc.correlated_values(popt, pcov)
    print("\n---Exponential fit---\n")
    print("R^2 = ", Rsq, "\n")
    print('fit parametes: a=', ae, 'b=', be, 'c=', ce)
    return popt, pcov, Rsq


def index_of_max_R2(df, country):
    R2_values = []
    start = 15
    for i in range(start, df[country].dropna().shape[0]):
        R2 = fit_exponential(country, df, [0, i])[2]
        R2_values.append(R2)
    print(R2_values)
    return start + R2_values.index(max(R2_values))


def fit_logistic(country, df):
    firstday = 0
    lastday = df[country].dropna().shape[0]
    xdata = df['Day'][(df['Day'] >= firstday) & (df['Day'] < lastday)]
    ydata = df[country][(df['Day'] >= firstday) & (df['Day'] < lastday)]
    popt, pcov = curve_fit(logistic, xdata, ydata, bounds=(
        [0, 0, 0, -1e2, 0], [1e6, 1e5, 1, 1e2, 1e2]), maxfev=1e6)
    x = np.linspace(firstday, lastday + 10, 100)
    y = logistic(x, *popt)
    inf_p = inflection_point(x, y)
    al, bl, cl, dl, el = unc.correlated_values(popt, pcov)
    print("\n---Logistic fit---\n")
    print("R^2 = ", R2(ydata, logistic(xdata, *popt)), "\n")
    print('fit parametes: a=', al, 'b=', bl, 'c=', cl, 'd=', dl, 'e=', el)
    return popt, pcov, inf_p


def inflection_point(x, y):
    dy = np.diff(y)
    idx_max_dy = np.argmax(dy)
    return idx_max_dy, x[idx_max_dy]


def plot_fits(country, df, exp_popt, exp_pcov, log_popt, log_pcov, inflection_p_idx, case):
    plt.figure(figsize=(10, 5))
    firstday = 0
    lastday = df[country].dropna().shape[0]
    xdata = df['Day'][(df['Day'] >= firstday) & (df['Day'] < lastday)]
    ydata = df[country][(df['Day'] >= firstday) & (df['Day'] < lastday)]
    x = np.linspace(firstday, lastday + 10, 100)
    plt.plot(xdata, ydata, 'k.', label='data')
    plt.plot(x, exponential(x, *exp_popt), 'r-', label="Exponential fit")
    plt.plot(x, logistic(x, *log_popt), 'b-', label='Logistic fit')
    ae, be, ce = unc.correlated_values(exp_popt, exp_pcov)
    py = ae * unp.exp(be * x) + ce
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)
    plt.fill_between(x, nom + 2 * std, nom - 2 *
                     std, facecolor="red", alpha=.2)
    al, bl, cl, dl, el = unc.correlated_values(log_popt, log_pcov)
    py = al / (1 + bl * unp.exp(-cl * (x - dl))) + el
    nom = unp.nominal_values(py)
    std = unp.std_devs(py)
    plt.fill_between(x, nom + 2 * std, nom - 2 *
                     std, facecolor="blue", alpha=.2)
    plt.plot(x[inflection_p_idx], logistic(x, *log_popt)
             [inflection_p_idx], 'oy', label='inflection point')
    plt.legend()
    if case == "confirmed":
        set_cases_labels()
    elif case == "deaths":
        set_deaths_labels()
    plt.ylim(-ydata.max() / 10., 2 * ydata.max())
    plt.title(country, loc='center')
    plt.show()


def add_daily_entries(df):
    df_daily_entries = df.copy()
    for country in df.columns:
        if country != "Date" and country != "Day":
            daily_column = np.append([0], df[country].iloc[0:-1])
            df_daily_entries["daily " + country] = df[country] - daily_column
    return df_daily_entries


def fit_normal(df, country):
    x = np.linspace(0, df["daily " + country].dropna().shape[0],
                    df["daily " + country].dropna().shape[0])
    y = df["daily " + country].dropna()
    popt, pcov = curve_fit(
        gauss, x, y, [df["daily " + country].max(), 100, 10])
    print(popt)
    print("covariance matrix")
    print(pcov)
    plt.plot(df["daily " + country], '.k')
    x = np.linspace(0, df["daily " + country].shape[0] *
                    2, df["daily " + country].shape[0] * 2)
    perr = np.sqrt(np.diag(pcov))  # standard errors
    plt.plot(x, gauss(x, *popt + perr), 'g--')
    plt.plot(x, gauss(x, *popt - perr), 'g--')
    plt.plot(x, gauss(x, *popt), 'r-',
             label='fit: a=%5.3f, x0=%5.3f, sigma=%5.3f' % tuple(popt))
    plt.xlim([0, df["daily " + country].shape[0] * 2])


def plot_daily_vs_total(df, country, interval):
    daily_column = np.append([0], df[country].iloc[0:-1])
    df["daily"] = df[country] - daily_column
    x, y = [], []
    for i in range(interval, df[country].shape[0], interval):
        x.append(df[country].iloc[i - 1])
        y.append(df["daily"].iloc[i - interval:i].sum())
    plt.plot(x, y, ".-", label=country)
    plt.xlabel("number of total cases")
    plt.ylabel("number of daily cases")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()


def plot_top_countries(df, countries, case):
    plt.figure(figsize=(15, 7))
    countries = countries.tolist()
    countries.append("Greece")

    data_bar_cases = []
    for country in countries:
        data_bar_cases.append(df[country].dropna().iloc[-1])
    if case == "confirmed":
        plt.ylabel("Number of confirmed cases")
    elif case == "deaths":
        plt.ylabel("Number of deaths")

    x = np.arange(11)
    bars = plt.bar(x, data_bar_cases)
    plt.xticks(x, countries, rotation=45)
    plt.xlabel("Country")
    plt.gcf().subplots_adjust(bottom=0.3)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, yval)


def top_countries(df):
    df_grouped = df.groupby(by='Country/Region', as_index=False).agg('sum')
    top_countries = df_grouped.nlargest(10, df.columns[-1])['Country/Region']
    return top_countries


def print_mortality_rates(df_cases, df_deaths, top_countries):
    print("Mortality rates for the countries with the highest number of deaths")
    print("-------------------------------------------------------------------")
    for country in top_countries:
        print(country,
              "(", df_deaths[country].dropna().iloc[-1], " deaths ): ",
              round(float(df_deaths[country].dropna().iloc[-1]) / float(df_cases[country].dropna().iloc[-1]) * 100, 1), "%")


def print_percentage_infected(df, df_population, top_countries):
    print("\nPopulation percentage infected")
    print("--------------------------------")
    for country in top_countries:
        country_pop = country
        if country == "US":
            country_pop = "United States"
        if country == "Iran":
            country_pop = "Iran, Islamic Rep."
        if country == "Korea, South":
            country_pop = "Korea, Rep."
        print(country,
              ": ", round(float(df[country].iloc[-1]) / float(df_population[df_population["Country Name"] == country_pop]["2018"]) * 100, 3), "%")


def print_permil_deaths(df, df_population, top_countries):
    print("\nPopulation permil dead")
    print("------------------------")
    for country in top_countries:
        country_pop = country
        if country == "US":
            country_pop = "United States"
        if country == "Iran":
            country_pop = "Iran, Islamic Rep."
        if country == "Korea, South":
            country_pop = "Korea, Rep."
        print(country, ": ", round(float(df[country].iloc[-1]) / float(
            df_population[df_population["Country Name"] == country_pop]["2018"]) * 1000, 5), "permil")
