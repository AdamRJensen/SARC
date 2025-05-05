
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import pvlib
import os

# temperature air
# Add calculation of CHP1
# Calculate DNI from Hukseflux & SPN1 & decomposition models
# Correct LWD calculation
# Add date specific import

# %%


# This probably doesn't work for duplicated time stamps
def asfreq_with_report(df, freq, round_value=None, report=False):
    org_index = df.index
    df_new = df
    if round_value is not None:
        df_new.index = df.index.round(round_value)
    df_new = df_new.asfreq(freq)
    new_index = df_new.index

    duplicated = org_index[org_index.duplicated()]
    removed = org_index.difference(new_index)
    added = new_index.difference(org_index)
    if report:
        print("\nTime stamps info")
        print(f"Duplicated: {len(duplicated)}")
        print(f"Errorneous: {org_index.isna().sum()}")
        print(f"Removed:    {len(removed)}")
        print(f"Added:      {len(added)}")
    return df_new


def pt_temperature(resistance, ohms_at_zero_c=100):
    alpha = 3.9080 * 10**-3
    beta = -5.8019 * 10**-7
    T = (- alpha + np.sqrt(alpha**2 - 4 * beta * (- resistance / ohms_at_zero_c + 1)))/ (2 * beta)
    return T


def calc_longwave_downwelling_irradiance(milivolt, sensitivity, temperature):
    temperature_kelvin = temperature + 273.15
    stefan_boltzmann_constant = 5.67 * 10**-8
    lwd = milivolt / sensitivity + stefan_boltzmann_constant * temperature_kelvin**4
    return lwd


column_rename_dict = {
    # Commented out entries means that they were changed multiple times
    # Naming changed on 2025-03-26 23:59:00+00:00
    'Lufft_WS601_precip_intensive_mmh': 'Lufft_WS601_precipitation_intensity_mmh',
    # 'Lufft_WS601_relative_pressure_hPa': 'Lufft_WS601_air_pressure_relative_hPa',
    # 'Lufft_WS601_abs_air_pressure_hPa': 'Lufft_WS601_air_pressure_absolute_hPa',
    'Lufft_WS601_relative_humidity_per': 'Lufft_WS601_humidity_relative_per',
    # 'Lufft_WS601_precip_absolute_mm': 'Lufft_WS601_precipitation_absolute_mm',
    # 'Lufft_WS601_precip_different_mm': 'Lufft_WS601_precipitation_difference_mm',
    # Naming changed on 2025-04-28 13:05
    'MS80SH_S24053407_out_voltage_mV': 'MS80SH_S24053407_GHI_mV',
    'DR30_65086_out_voltage_mV': 'DR30_65086_DNI_mV',
    'SR30_23485_out_voltage_mV': 'SR30_23485_GHI_mV',
    'SR300_45389_out_voltage_mV': 'SR300_45389_GHI_mV',
    'SP522_1246_out_voltage_mV': 'SP522_1246_GHI_mV',
    'SP422_1843_out_voltage_mV': 'SP422_1843_GHI_mV',
    'Lufft_WS601_precipitation_difference_mm': 'Lufft_WS601_precipitation_mm',
    'Lufft_WS601_precipitation_absolute_mm': 'Lufft_WS601_precipitation_cumulative_mm',
    'SP522_1246_GHI_Wm2': 'SP522_1265_GHI_Wm2',
    'SP522_1246_GHI_mV': 'SP522_1265_GHI_mV',
    'SP522_1246_heater_state': 'SP522_1265_heater_state',
    # Naming changed on 2025-05-02
    'Lufft_WS601_precip_type': 'Lufft_WS601_precipitation_type',
    'Lufft_WS601_air_temperature_degC': 'Lufft_WS601_temperature_air_degC',
    'Lufft_WS601_precip_absolute_mm': 'Lufft_WS601_precipitation_cumulative_mm',
    'Lufft_WS601_precip_different_mm': 'Lufft_WS601_precipitation_mm',
    'Lufft_WS601_precip_intensive_mmh': 'Lufft_WS601_precipitation_intensity_mmh',
    'Lufft_WS601_relative_humidity_per': 'Lufft_WS601_humidity_relative_per',
    'Lufft_WS601_relative_pressure_hPa': 'Lufft_WS601_pressure_relative_air_hPa',
    'Lufft_WS601_air_pressure_relative_hPa': 'Lufft_WS601_pressure_relative_air_hPa',
    'Lufft_WS601_abs_air_pressure_hPa': 'Lufft_WS601_pressure_absolute_air_hPa',
    'Lufft_WS601_air_pressure_absolute_hPa': 'Lufft_WS601_pressure_absolute_air_hPa',
}


def get_file_date(file):
    return pd.to_datetime(os.path.basename(file)[:19], format='%Y-%m-%d_%H-%M-%S')


def select_files_from_period(files, start, end):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    files = [f for f in files if (get_file_date(f) >= start) & (get_file_date(f) <= end)]
    return files


# %%
path = 'C:/Users/arajen/Downloads/station_data/'

start = '2025-05-03'
end = '2025-05-05'

na_values = [
    "NAN",  # crbasic convention for floats
    -7999,  # crbasic convention for integers (FP2)
    -2147483648,  # crbasic convetion for integers (IEEE4)
    -214748400,  # Lufft weather station and EKO rotation shadow band
    -21474840,  # Seen on DR30_65086_rh_per
    -3579140.0,  # Seen on SR30_23485
    # -715194,  # SR30 (only one occurence)
]

dfs = []
for speed in ['Fast', 'Slow']:
    for level in ['Primary', 'Secondary']:
        files_all = glob.glob(path + f"*{level}_{speed}_table.csv*")
        files = select_files_from_period(files_all, start, end)
        dfis = [
            pd.read_csv(
                f, skiprows=[0, 2, 3], index_col=0,
                parse_dates=[0], na_values=na_values)
            for f in files]
        dfis = [d.rename(columns=column_rename_dict) for d in dfis]
        dfi = pd.concat(dfis, axis='rows').sort_index()
        # Round to nearest second (due to inclusion of microseconds)
        dfi.index = dfi.index.round('1s')
        # Duplicated rows needs to be removed before concatting
        duplicated_rows = dfi.index.duplicated(keep='first')
        if duplicated_rows.sum() > 0:
            print(f"Duplicated rows: {duplicated_rows.sum()}")
            dfi = dfi[~duplicated_rows]
        dfs.append(dfi)

df_raw = pd.concat(dfs, axis='columns')

df = asfreq_with_report(df_raw, freq='1s', round_value='1s', report=True)

df = df.resample('1min').mean()

# %%

df_raw['SHP1_185163_DNI_Wm2'].dropna().index.diff().value_counts()

# %%
df['StarSchenk_7773_GHI_Wm2'] = 83.8 * df['StarSchenk_7773_GHI_mV']
df['Licor_PY116375_GHI_Wm2'] = 100 * df['Licor_PY116375_GHI_mV']
df['CMP11_128758_DHI_Wm2'] = df['CMP11_128758_DHI_mV'] / (9.89 * 10**-3)
df['CMP11_128767_GHI_Wm2'] = df['CMP11_128767_GHI_mV'] / (8.02 * 10**-3)
df['CHP1_140049_DNI_Wm2'] = df['CHP1_140049_DNI_mV'] / (7.87 * 10**-3)
df['CGR4_170223_LWD_Wm2'] = (
    df['CGR4_170223_LWD_mV'] / (8.95 * 10**-3)
    + 5.67*10**-8 * (df['CGR4_170223_temperature_degC'] + 273.15)**4)

df['CGR4_170223_temperature_calc_degC'] = pt_temperature(
    resistance=df['CGR4_170223_temperature_ohm'], ohms_at_zero_c=100)
df['CHP1_140049_temperature_calc_degC'] = pt_temperature(
    resistance=df['CHP1_140049_temperature_ohm'], ohms_at_zero_c=100)

CHP1_140049_temperature_depency = {
    50: 0.16, 40: -0.02, 30: -0.07, 20: 0.0, 10: 0.18, 0: 0.19, -10: 0.05, -20: -0.48}  # [%]

# %% Calculate solar position

location_dtu = pvlib.location.Location(
    latitude=55.79064, longitude=12.52505, altitude=50)

solpos = location_dtu.get_solarposition(df.index)

df['Calc_from-DHI-DNI_GHI_Wm2'] = \
    df['SMP22_200060_DHI_Wm2'] + \
    df['SHP1_185163_DNI_Wm2'] \
    * np.clip(np.cos(np.deg2rad(solpos['apparent_zenith'])), a_min=0, a_max=None)
df['Calc_from-GHI-DNI_DHI_Wm2'] = \
    df['SMP22_200057_GHI_Wm2'] - \
    df['SHP1_185163_DNI_Wm2'] \
    * np.clip(np.cos(np.deg2rad(solpos['apparent_zenith'])), a_min=0, a_max=None)
df['Calc_from-GHI-DHI_DNI_Wm2'] = pvlib.irradiance.complete_irradiance(
    solpos['apparent_zenith'], ghi=df['SMP22_200057_GHI_Wm2'], dhi=df['SMP22_200060_DHI_Wm2'])['dni']
df['SPN1_A270_DNI_Wm2'] = pvlib.irradiance.complete_irradiance(
    solpos['apparent_zenith'], ghi=df['SPN1_A270_GHI_Wm2'], dhi=df['SPN1_A270_DHI_Wm2'])['dni']


# %%


def convert_parameters_to_table(sensor_names):
    meta = {}
    for sensor in sensor_names:
        split = sensor.split('_')
        meta[sensor] = {}
        meta[sensor]['sensor'] = split[0]
        meta[sensor]['serial_number'] = split[1]
        meta[sensor]['parameter'] = split[2]
        if len(split) > 4:
            modifier = split[3]
        else:
            modifier = ''
        meta[sensor]['parameter_modifier'] = modifier
        meta[sensor]['unit'] = split[-1]
    return meta


meta = pd.DataFrame(convert_parameters_to_table(df.columns)).T

# %%
fig, axes = plt.subplots(nrows=2, sharex=True)

freq = '1min'
solpos['apparent_zenith'].resample(freq).mean().plot(ax=axes[0], linestyle='--', label='Solar position')
df['Solys2_140027_tilt_deg'].resample(freq).mean().plot(ax=axes[0], label='Tracker position')

solpos['azimuth'].resample(freq).mean().plot(ax=axes[1], linestyle='--', alpha=0.8, label='Solar position')
df['Solys2_140027_azimuth_deg'].resample(freq).mean().plot(ax=axes[1], alpha=0.8, label='Tracker position')


axes[0].legend(ncol=2, loc='upper center', bbox_to_anchor=[0.5, 1.3])
axes[-1].set_xlim(None, None)
axes[-1].set_xlabel(None)
axes[0].set_ylabel('Zenith/tilt [°]')
axes[1].set_ylabel('Azimuth [°]')
axes[1].set_ylim(0, 360)
axes[1].set_yticks([0, 90, 180, 270, 360])
axes[0].set_ylim(axes[0].get_ylim())
axes[0].fill_between(axes[0].get_xlim(), 90, axes[0].get_ylim()[1], color='lightgrey')


# %% Plots for each parameter and unit
for m in meta['parameter'].unique():
    for unit in meta.loc[meta['parameter']==m, 'unit'].unique():
        fig, ax = plt.subplots()
        ax.set_title(f"{m} [{unit}]")
        for i in meta[(meta['parameter'] == m) & (meta['unit'] == unit)].index:
            df[i].plot(ax=ax, label=' '.join(i.split('_')[:2]))
        ax.legend()
        plt.show()

# %% Plots for each sensor

for s in meta['sensor'].unique():
    axes = df[meta[meta['sensor'] == s].index].iloc[-60*20:].plot(sharex=True, subplots=True, figsize=(8, 8))
    axes[0].set_title(s)
    plt.show()

# %% Plotting function


def plot_reference(data, reference, parameters, ncols=3, figsize=(8, 8),
                   title=None, xlim=(-10, 600), ylim=None, kind='relative'):

    if ylim is None:
        if kind == 'relative':
            ylim = (0.5, 1.5)
        elif kind == 'difference':
            ylim = (-50, 50)
        else:
            ylim = xlim

    fig, axes = plt.subplots(
        ncols=ncols, nrows=int(np.ceil(len(parameters)/3)),
        sharex=True, sharey=True, figsize=figsize)

    params = {'s': 1, 'zorder': 10}
    for parameter, ax in zip(parameters, axes.flatten()):
        if kind == 'relative':
            ax.scatter(data[reference], data[parameter]/data[reference], **params)
        elif kind == 'difference':
            ax.scatter(data[reference], data[parameter]-data[reference], **params)
        else:
            ax.scatter(data[reference], data[parameter], **params)
        ax.set_title(parameter)
        ax.grid(alpha=0.4, zorder=-1)

    if kind != 'relative':
        pass
    elif kind == 'difference':
        pass
    else:
        ax.plot(xlim, ylim, c='r', alpha=0.5, lw=1, zorder=-1)
        ax.set_aspect('equal')

    axes[0, 0].set_xlim(xlim)
    axes[0, 0].set_ylim(ylim)
    fig.suptitle(title)
    fig.tight_layout()


# %%
component = 'DNI'

reference = {
    'GHI': 'Calc_from-DHI-DNI_GHI_Wm2',
    'DHI': 'SMP22_200060_DHI_Wm2',
    'DNI': 'DR30_65086_DNI_Wm2',
}

xlim_upper = {'GHI': 1000, 'DHI': 500, 'DNI': 1000}

sensors = meta[
    (meta['parameter'] == component) &
    (meta['unit'] == 'Wm2') &
    (meta['parameter_modifier'] != 'raw')
].index

for kind in ['relative', 'difference', 'normal']:
    plot_reference(df, reference[component], sensors, title=component,
                   kind=kind, xlim=(0, xlim_upper[component]))

# %% Compare GHI

# fig, axes = plt.subplots(ncols=4, nrows=int(np.ceil(len(ghi_sensors)/4)),
#                          sharex=True, sharey=True, figsize=(8, 8))
# for sensor, ax in zip(ghi_sensors, axes.flatten()):
#     ax.scatter(df['Calc_from-DHI-DNI_GHI_Wm2'], df[sensor], s=1, zorder=10)
#     ax.set_title('_'.join(sensor.split('_')[:2]))
#     ax.set_aspect('equal')
#     ax.grid(alpha=0.4, zorder=-1)
#     ax.plot([0, 1000], [0, 1000], c='r', alpha=0.5, lw=1, zorder=-1)

# axes[0, 0].set_ylim(-10, 1100)
# axes[0, 0].set_xlim(-10, 1100)
# fig.suptitle('GHI')

# fig.tight_layout()

# %%
# df[df[ghi_sensors]<-100] = np.nan
# dfp = df[ghi_sensors].subtract(df['Calc_from-DHI-DNI_GHI_Wm2'], axis='rows')
# ax = dfp.plot(xlim=('2025-04-01 08','2025-04-01 16'), ylim=(-40,40), legend=True)
dfp = df[dhi_sensors]

# ax = dfp.plot(xlim=('2025-04-01 08','2025-04-01 16'), ylim=(300,700), legend=True)
xlim = ('2025-04-03 04', '2025-04-06 18')
xlim = ('2025-04-12 04', '2025-04-16 18')
ax = dfp.plot(xlim=xlim, ylim=(0, 700), legend=True)
ax.legend(ncol=2, loc='upper center', bbox_to_anchor=[0.5, 1.55])


# %%
ax=df.loc[solpos['elevation'] <- 10, ghi_sensors].plot.hist(
    bins=61, range=(-3, 3), histtype='step')
ax.legend(ncol=2, loc='upper center', bbox_to_anchor=[0.5, 1.55])

# %% Spikes in DNI
delta = pd.Timedelta(seconds=15)
spike_instances = df.index[df['SHP1_185163_DNI_raw_Wm2'].isna().astype(int).diff() == -1]
for spike in spike_instances:
    df.loc[spike - delta: spike + delta, 'SHP1_185163_DNI_raw_Wm2'].plot(marker='o')
    plt.show()

# %%
notkipp = [c for c in df.columns if c[:3] not in
           ['CMP', 'CHP', 'SHP', 'SMP', 'Sol', 'Luf', 'CGR', 'CVF', 'SGR', 'PH1', 'PR1']]

fig, axes = plt.subplots(nrows=len(notkipp), sharex=True, figsize=(6, 25))
for col, ax in zip(notkipp, axes):
    ax.plot(df.loc['2025-04-09 15': '2025-04-10 12', col])
    ax.set_ylabel(col[:7])

# df[notkipp].plot(xlim=['2025-04-09 15','2025-04-10 12'], sharex=True, figsize=(8,16), subplots=True)
