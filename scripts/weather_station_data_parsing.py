
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import pvlib
import os

# %%
# Better calculate Calc_from-GHI-DNI_DHI and Calc_from-DHI-DNI_GHI (they have spikes)
# Calculate bias, rmse (for elevation above > 5deg)
# Create bias/rmse function


# Assess duplicate columns
# dsdf.columns[df.columns.duplicated(keep=False)]

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


def calculate_metrics(reference, test, condition=None):
    if len(reference) != len(test):
        raise ValueError("Reference and test data does not have the same length.")
    if condition is None:
        condition = np.ones(len(reference)).astype(bool)
    metrics = {
        'rmse': np.sqrt(np.mean((test[condition] - reference[condition])**2)),
        'bias': np.mean(test[condition] - reference[condition]),
    }
    reference_mean = np.mean(reference[condition])
    metrics['rmse_percent'] = metrics['rmse'] / reference_mean * 100
    metrics['bias_percent'] = metrics['bias'] / reference_mean * 100
    return metrics


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
    'SP522_1246_out_voltage_mV': 'SP522_1246_GHI_mV',  # removed 2025-08-11
    'SP422_1843_out_voltage_mV': 'SP422_1843_GHI_mV',
    'Lufft_WS601_precipitation_difference_mm': 'Lufft_WS601_precipitation_mm',
    'Lufft_WS601_precipitation_absolute_mm': 'Lufft_WS601_precipitation_cumulative_mm',
    'SP522_1246_GHI_Wm2': 'SP522_1265_GHI_Wm2',  # removed 2025-08-11
    'SP522_1246_GHI_mV': 'SP522_1265_GHI_mV',  # removed 2025-08-11
    'SP522_1246_heater_state': 'SP522_1265_heater_state',  # removed 2025-08-11
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
    # Unknown date
    'SPN1_A270_Heater_ratio': 'SPN1_A270_heater_ratio',
    'SPN1_A270_Sun_ratio': 'SPN1_A270_sun_ratio',
}


def get_file_date(file):
    return pd.to_datetime(os.path.basename(file)[:19], format='%Y-%m-%d_%H-%M-%S')


def select_files_from_period(files, start=None, end=None, days_offset=1):
    offset = pd.Timedelta(days=days_offset)  # due to file naming convention
    if start is not None:
        start = pd.Timestamp(start) + offset
        files = [f for f in files if get_file_date(f) >= start]
    if end is not None:
        end = pd.Timestamp(end) + offset
        files = [f for f in files if get_file_date(f) <= end]
    return files


def combined_columns(df, column):
    if column in df.columns[df.columns.duplicated()]:
        temp_c = df[column].copy()
        del df[column]
        df[column] = temp_c.iloc[:, 0].values
        df.loc[df[column].isna(), column] = temp_c.iloc[:, 1]
    return df

# %% Read data
path = 'C:/Users/arajen/Downloads/station_data/'

start = pd.Timestamp.today() - pd.Timedelta(days=4)
end = pd.Timestamp.today()
#start = '2025-08-01'
#end = '2025-08-30'

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

df_raw = pd.concat(dfs, axis='columns', join='outer')

# Resolve duplicated column names
# This was caused by moving the parameter from fast to slow scan
df_raw = combined_columns(df_raw, 'DR30_65086_temperature_degC')
df_raw = combined_columns(df_raw, 'SR30_23485_temperature_degC')

df = asfreq_with_report(df_raw, freq='1s', round_value='1s', report=True)

df = df.resample('1min').mean()


# %% Corrections
df.loc['2025-04-06 12:00':'2025-04-07 23:59', 'StarSchenk_7773_GHI_mV'] = np.nan
df.loc['2025-06-03 10:00':'2025-06-17 14:00', 'StarSchenk_7773_GHI_mV'] = np.nan
df.loc['2025-05-23 10:00':'2025-05-24 17:00', 'StarSchenk_7773_GHI_mV'] = np.nan

df.loc[df['SPN1_A270_heater_ratio']==-524288, 'SPN1_A270_heater_ratio'] = np.nan


# Tracker was turned off
tracker_instruments = ['CGR4_170223', 'DR30_65086', 'CHP1_140049', 'SMP12_233555', 'SMP22_200060', 'SHP1_185163', 'SMP10_196704']
tracker_parameters = [c for c in df.columns if '_'.join(c.split('_')[:2]) in tracker_instruments]
df.loc['2025-07-10 09:15': '2025-07-10 10:40', tracker_parameters] = np.nan

# Incorrect wiring
df.loc['2025-07-04 14:20': '2025-07-09 23:59', [c for c in df.columns if c.startswith('SMP12')]] = np.nan

# Licor bubble level was shut
df.loc['2025-06-30 13:25': '2025-07-10 13:32', [c for c in df.columns if c.startswith('Licor')]] = np.nan

# tracker shadow ball was off
# df.loc['2025-06-30 14:00': '2025-07-03 15:00', ['SMP22_200060_GHI_Wm2', 'SMP22_200060_GHI_raw_Wm2']] = \
df[['SMP22_200060_GHI_Wm2', 'SMP22_200060_GHI_raw_Wm2']] = \
    df.loc['2025-06-30 14:00': '2025-07-03 15:00', ['SMP22_200060_DHI_Wm2', 'SMP22_200060_DHI_raw_Wm2']]  # .values

df.loc['2025-06-30 12:00': '2025-07-03 15:00',
        ['SMP22_200060_DHI_Wm2', 'SMP22_200060_DHI_raw_Wm2', 'SMP22_200060_temperature_degC']] = np.nan

# SHP1 logged all zeros
df.loc['2025-09-08 12:38':'2025-09-10 10', ['SHP1_185163_DNI_Wm2','SHP1_185163_DNI_raw_Wm2','SHP1_185163_temperature_degC']] = np.nan

# Delete old MS80SHplus columns
try:
    del df[[c for c in df.columns if c.startswith('MS80SHplus_1209')]]
except pd.errors.InvalidIndexError:
    pass

# df = df.rename(columns={
#     'CMP11_128758_DHI_mV': 'Shadow-band_128758_DHI_mV',
#     'CMP11_128758_DHI_Wm2': 'Shadow-band_128758_DHI_Wm2',
# })

# XXX end-date should be adjusted
# df.loc['2025-05-23':, 'SP522_1265_GHI_Wm2'] = np.nan

# %%
df['StarSchenk_7773_GHI_Wm2'] = 83.8 * df['StarSchenk_7773_GHI_mV']
df['Licor_PY116375_GHI_Wm2'] = 100 * df['Licor_PY116375_GHI_mV']


df['SP510_3882_GHI_Wm2'] = 22.88 * df['SP510_3882_GHI_mV']

df['CMP11_128758_DHI_Wm2'] = df['CMP11_128758_DHI_mV'] / (9.89 * 10**-3)
df['CMP11_128767_GHI_Wm2'] = df['CMP11_128767_GHI_mV'] / (8.02 * 10**-3)
df['CHP1_140049_DNI_Wm2'] = df['CHP1_140049_DNI_mV'] / (7.87 * 10**-3)
df['CGR4_170223_LWN_Wm2'] = df['CGR4_170223_LWD_mV'] / (8.95 * 10**-3)
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

location = pvlib.location.Location(
    latitude=55.79064, longitude=12.52505, altitude=50)

solpos = location.get_solarposition(df.index)

df['solar_zenith'] = solpos['apparent_zenith']
df['solar_elevation'] = solpos['apparent_elevation']
df['solar_azimuth'] = solpos['azimuth']

# %% Clear sky
# cams, _ = pvlib.iotools.get_cams(
#     location.latitude, location.longitude,
#     start=df.index.min(), end=df.index.max(),
#     email='arajen@dtu.dk', time_step='1min')

# cams = cams.reindex(df.index)

# df['is_clear'] = (df['DR30_65086_DNI_Wm2'] / cams['dni_clear']) > 0.9
# df['is_clear'] = df['is_clear'].rolling('30min').min()
# df['is_overcast'] = df['DR30_65086_DNI_Wm2'] < 5

# %%


df_sub = df.copy()

df_sub = df_sub.rename(columns={
    'DR30_65086_DNI_Wm2': 'DNI',
    'SR300_45389_GHI_Wm2': 'GHI',
    'SMP22_200060_DHI_Wm2': 'DHI'})


ax = df_sub.loc['2025-08-18', ['GHI','DHI','DNI']].plot()
ax.set_xlabel('')
ax.set_ylabel('Irradiance [W/m$^2$]')
ax.set_title('Clear-sky day')
ax.set_ylim(0, 1000)

# %%
zenith_threshold = 87

df['mu'] = np.cos(np.deg2rad(solpos['apparent_zenith'])).clip(lower=0)

# df['BHI'] = (df['SHP1_185163_DNI_Wm2'] * df['mu']).clip(lower=0)

# df['GHI_diff'] = df['SMP22_200057_GHI_Wm2'] - df['Calc_from-DHI-DNI_GHI_Wm2']

# df['GHI_diff_rel'] = (df['SMP22_200057_GHI_Wm2'] / df['Calc_from-DHI-DNI_GHI_Wm2'])*100 - 100

# df['DNI_diff'] = df['SHP1_185163_DNI_Wm2'] - df['DR30_65086_DNI_Wm2']


df['Calc_from-DHI-DNI_GHI_Wm2'] = (
    df['SMP22_200060_DHI_Wm2'] + df['SHP1_185163_DNI_Wm2'] * df['mu']).clip(lower=0)
#    df['SMP22_200060_DHI_Wm2'] + df['DR30_65086_DNI_Wm2'] * df['mu']).clip(lower=0)
# df.loc[df['Calc_from-DHI-DNI_GHI_Wm2'] > 1500, 'Calc_from-DHI-DNI_GHI_Wm2'] = np.nan

df['Calc_from-GHI-DNI_DHI_Wm2'] = (
    df['SMP22_200057_GHI_Wm2'] - df['SHP1_185163_DNI_Wm2'] * df['mu']).clip(lower=0)

df['Calc_from-GHI-DHI_DNI_Wm2'] = (
    df['SMP22_200057_GHI_Wm2'] - df['SMP22_200060_DHI_Wm2']).clip(lower=0) / df['mu']
df.loc[solpos['apparent_zenith'] > zenith_threshold, 'Calc_from-GHI-DHI_DNI_Wm2'] = np.nan

df['SPN1_A270_DNI_Wm2'] = pvlib.irradiance.complete_irradiance(
    solpos['apparent_zenith'], ghi=df['SPN1_A270_GHI_Wm2'], dhi=df['SPN1_A270_DHI_Wm2'])['dni']

df['SR_Calc-SR300-SRD100_DNI_Wm2'] = (
    df['SR300_45389_GHI_Wm2'] - df['SRD100_14401_DHI_Wm2']).clip(lower=0) / df['mu']
df.loc[solpos['apparent_zenith'] > zenith_threshold, 'SR_Calc-SR300-SRD100_DNI_Wm2'] = np.nan


# df['MS80plus_S24067011_DNI_calc_Wm2'] = pvlib.irradiance.complete_irradiance(
#     solpos['apparent_zenith'], ghi=df['MS80plus_S24067011_GHI_Wm2'], dhi=df['MS80plus_S24067011_DHI_Wm2'])['dni']
df['MS80plus_S24067011_DNI_Wm2'] = pvlib.irradiance.complete_irradiance(
    solpos['apparent_zenith'], ghi=df['MS80plus_S24067011_GHI_Wm2'], dhi=df['MS80plus_S24067011_DHI_Wm2'])['dni']

df['SMP12_233555_tilt_calc_deg'] = np.sqrt(df['SMP12_233555_pitch_deg']**2 + df['SMP12_233555_roll_deg']**2)



# %% Remove empty columns

for c in df.columns:
    if df[c].empty | df[c].isna().all():
        print(f"Deleting empty column: {c}")
        del df[c]

# %%


def convert_parameters_to_table(sensor_names):
    meta = {}
    for sensor in sensor_names:
        split = sensor.split('_')
        if len(split) < 4:  # ignore, e.g., solar_zenith
            continue
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

meta = meta.sort_values('parameter')

# meta = meta[meta['sensor'] != 'MS80SHplus']
# meta = meta[meta['sensor'] != 'StarSchenk']

meta['sensor_serial'] = meta['sensor'] + '_' + meta['serial_number']


# %%

razon = [c for c in df.columns if c[:2] in ['PH', 'PR', 'Ra']]

# %% Plots for each sensor

for s in meta['sensor_serial'].unique():
    axes = df[meta[meta['sensor_serial'] == s].index].plot(
        sharex=True, subplots=True, figsize=(8, 8))
    axes[0].set_title(s)
    for ax in axes:
        ax.legend(loc='upper left')
    axes[-1].set_xlabel(None)
    plt.show()



# %%

df['dif'] = df['SMP22_200060_DHI_Wm2'] - df['SMP22_200057_GHI_Wm2']
df['rel_dif'] = (df['SMP22_200060_DHI_Wm2'] - df['SMP22_200057_GHI_Wm2'])/df['SMP22_200057_GHI_Wm2']

df[df['solar_zenith']<85]['2025-09-09':].plot.scatter(x='DR30_65086_DNI_Wm2', y='dif', s=1, alpha=0.3, ylim=[-10,10])

# %% Plots for each parameter and unit
for m in meta['parameter'].unique():
    for unit in meta.loc[meta['parameter'] == m, 'unit'].unique():
        fig, ax = plt.subplots()
        ax.set_title(f"{m} [{unit}]")
        for i in meta[(meta['parameter'] == m) & (meta['unit'] == unit)].index:
            df[i].plot(ax=ax, label=' '.join(i.split('_')[:2]))
        ax.legend()
        plt.show()

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



# %% Plotting function


def plot_reference(data, reference, parameters, metric_condition=None,
                   c='b', ncols=3, figsize=None, title=None, xlim=(-10, 600),
                   ylim=None, kind='relative', **kwargs):
    # Remove reference from parameters
    parameters = [p for p in parameters if p != reference]

    nrows = int(np.ceil(len(parameters)/3))

    figsize = (ncols*2, nrows*2) if figsize is None else figsize

    if ylim is None:
        if kind == 'relative':
            ylim = (0.5, 1.5)
        elif kind == 'difference':
            ylim = (-100, 100)
        else:
            ylim = xlim

    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows,
        sharex=True, sharey=True, figsize=figsize)

    params = {'s': 1, 'zorder': 10}
    for parameter, ax in zip(parameters, axes.flatten()):
        metrics = calculate_metrics(data[reference], data[parameter], metric_condition)

        if kind == 'relative':
            ax.scatter(data[reference], data[parameter]/data[reference],
                       c=c, **params, **kwargs)
        elif kind == 'difference':
            ax.scatter(data[reference], data[parameter]-data[reference],
                       c=c, **params, **kwargs)
        elif kind == 'elevation':
            ax.scatter(data['solar_elevation'], data[parameter]-data[reference],
                       c=c, **params, **kwargs)
            hb = ax.hexbin(
                x=dfp[y], y=dfp[x],
                gridsize=100,
                # bins='log',
                # bins=[0, 100, 200, 1000],
                cmap='viridis',
                mincnt=1,  # Min. no. points for there to be a color
                extent=(0, 1500, 0, 1500),  # (xmin, xmax, ymin, ymax)
                norm='linear',
                vmin=10,
                vmax=100,
            )
        else:
            ax.scatter(data[reference], data[parameter], c=c, **params,
                       alpha=0.1, **kwargs)
            ax.plot(xlim, ylim, c='r', alpha=0.5, lw=1, zorder=-1)
            ax.set_aspect('equal')
            lim_delta = max(xlim[1], ylim[1]) - min(xlim[0], ylim[0])
            tick_spacing = 100 if lim_delta < 500 else 250
            ax.set_xticks(np.arange(0, xlim[1], tick_spacing))
            ax.set_yticks(np.arange(0, ylim[1], tick_spacing))
        metrics_text = f"RMSE: {metrics['rmse_percent']:.1f}%\nBias: {metrics['bias_percent']:.1f}%"
        ax.text(0.99, 0.01, metrics_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.0'),
                zorder=10)

        ax.set_title('-'.join(parameter.split('_')[0:1] +
                              parameter.split('_')[3:-1]))
        ax.grid(alpha=0.4, zorder=-1)
        ax.tick_params(axis='x', labelrotation=45)

    # Hide unused subplots
    [ax.set_visible(False) for i, ax in enumerate(axes.flatten()) if i >= len(parameters)]

    axes[0, 0].set_xlim(xlim)
    axes[0, 0].set_ylim(ylim)
    fig.suptitle(title)
    #fig.tight_layout()
    return fig, axes


# %%
reference = {
    'GHI': 'Calc_from-DHI-DNI_GHI_Wm2',
    #'GHI': 'SMP22_200057_GHI_Wm2',
    'DHI': 'SMP22_200060_DHI_Wm2',
    #'DHI': 'Calc_from-GHI-DNI_DHI_Wm2',
    #'DNI': 'PH1_190116_DNI_Wm2',
    'DNI': 'DR30_65086_DNI_Wm2',
}

df['Kd'] = df['SMP22_200060_DHI_Wm2'] / df['SMP22_200057_GHI_Wm2']

df['is_allweather'] = True
df['is_cloudy'] = df['SHP1_185163_DNI_Wm2'] < 2
df['is_clear'] = df['SHP1_185163_DNI_Wm2'] > 200

xlim_upper = {'GHI': 1200, 'DHI': 400, 'DNI': 1200}

#sky = 'is_cloudy'
#sky = 'is_allweather'
sky = 'is_clear'
dfp = df[(df['solar_zenith'] < 85) & df[sky]]#.resample('5min').mean()

#dfp = dfp[:'2025-07-17']
#dfp = dfp['2025-07-25':]

for component in ['GHI']:#, 'DHI', 'DNI']:
    sensors = meta[
        (meta['parameter'] == component)
        & (meta['unit'] == 'Wm2')
        & (meta['parameter_modifier'] != 'raw')
        & (meta.index != reference)
    ].index

    fig, axes = plot_reference(
        dfp, reference[component], sensors,
        metric_condition=dfp['solar_zenith'] < 80,
        title=component,
        #c=dfp['SHP1_185163_DNI_Wm2'],
        c=dfp['solar_zenith'],
        kind='difference',
        ncols={'GHI': 4}.get(component, 3),
        xlim=(0, xlim_upper[component]),
        **{'alpha': 0.2},
        )
    for ax in axes.flatten():
        ax.set_xticks([0, 250, 500, 750, 1000])
    #     ticks = np.arange(0, xlim_upper[component]+0.01, 250)
    #     ax.set_xticks(ticks)
    #     ax.set_yticks(ticks)
        # ax.set_ylabel('Measured [W/m$^2$]')
    #axes[1, 1].set_xlabel('Reference [W/m$^2$]')
    #axes.flatten()[-1].set_title(r'GHI - DNI$\cdot$cos($\theta$)')
    fig.suptitle(f"{component} {sky}")
    fig.tight_layout()


# %%
x = 'SR300_45389_GHI_Wm2'
y = 'SMP10_248585_GHI_Wm2'

from matplotlib.colors import Normalize

#from matplotlib.colors import 


dfp = df[df['solar_zenith']<90]

fig, ax = plt.subplots(figsize=(8, 8))
hb = ax.hexbin(
    x=dfp[y], y=dfp[x],
    # gridsize=100,
    # bins='log',
    # bins=[0, 100, 200, 1000],
    # cmap='viridis',
    mincnt=1,  # Min. no. points for there to be a color
    # extent=(0, 1500, 0, 1500),  # (xmin, xmax, ymin, ymax)
    norm=Normalize(vmin=0, vmax=100),
    # vmin=10,
    # vmax=100,
)
ax.set_aspect('equal')

cbar = fig.colorbar(hb, ax=ax)
cbar.set_label('Counts per hexbin')

# Set axis labels and title
ax.set_title('Hexbin Plot of Point Density')
ticks = np.arange(0, 1400+0.01, 200)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel(x)
ax.set_ylabel(y)

# %%

# SRD100 - 100% calibrated for clear sky
# have multipliers for clear and grey skys that could improve a lot
# GHI 90 % 


# %% Export to Viktar
solarband = [c for c in df.columns if 'solarband' in c.lower()]
df.loc['2025-05-22': , solarband + ['SHP1_185163_DNI_Wm2', 'SMP22_200060_DHI_Wm2', 'SMP22_200057_GHI_Wm2', 'solar_zenith', 'solar_azimuth']].to_csv('solarband_c3_data.csv')


# %%
# April 6th, 15th (23, 24)

day = pd.Timestamp('2025-04-15 00:00:00+0000', tz='UTC')

ghi_sensors = [
    'SMP10_248585_GHI_Wm2',
    # 'CMP11_128767_GHI_Wm2',
    'SMP12_233555_GHI_Wm2',
    'SR300_45389_GHI_Wm2',
    'MS80SH_S24053407_GHI_Wm2',
]
# for day in pd.DatetimeIndex(df.index.date).unique().tz_localize('UTC'):
fig, axes = plt.subplots(ncols=len(ghi_sensors), figsize=(10, 4), sharey=True)
for sensor, ax in zip(ghi_sensors, axes):
    df[day: day + pd.Timedelta(hours=10)].plot.scatter(
        ax=ax,
        x='Calc_from-DHI-DNI_GHI_Wm2',
        y=sensor,
        s=1,
        zorder=5,
        )
    # ax.set_title(day.strftime('%d %b'))
    ax.set_ylim(-5, 700)
    ax.set_xlim(-5, 700)
    ax.set_aspect('equal')
    ax.set_xlabel('Reference GHI [W/m$^2$]')
    ax.set_ylabel('Measured GHI [W/m$^2$]')
    ax.set_title(sensor.split('_')[0])
    ax.grid(alpha=0.3, zorder=-2)
fig.tight_layout()
plt.show()

# %%
ax = df.loc[solpos['elevation'] <- 10, ghi_sensors].plot.hist(
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


# %%

nan_values = df.isna().resample('1d').mean()['2025-08-08':]

nan_values.columns[(nan_values!=0).any()]

# %% Tilt comparison


df[['solar_zenith', 'DR30_65086_tilt_deg', 'Solys2_140027_tilt_deg']].plot()
#'Solys2_140027_azimuth_deg', 'Solys2_140027_tilt_deg'
plt.show()


df.loc[df['solar_zenith']<89, ['DR30_65086_tilt_deg', 'Solys2_140027_tilt_deg']].subtract(df['solar_zenith'], axis='rows').plot(
    ylim=[-5,5])


# %%

df.loc['2025-09-08 12:50':, ['SMP22_200060_DHI_Wm2', 'SMP22_200057_GHI_Wm2']].diff(axis='columns').plot()

# %%

for d in df.index.to_series().dt.date.unique():
    mask = (df.index.date==d) & (df['DR30_65086_DNI_Wm2'] > 200)
    df_sub = df[mask].resample('5min').mean()
    df_sub[['DR30_65086_DNI_Wm2', 'SHP1_185163_DNI_Wm2', 'CHP1_140049_DNI_Wm2','PH1_190116_DNI_Wm2']].divide(df_sub['DR30_65086_DNI_Wm2'], axis=0).plot(ylim=[0.95,1.05])
    plt.show()
