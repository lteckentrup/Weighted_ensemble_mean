import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import xskillscore as xs
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
import seaborn as sns

def readin(method,model,var):
    if method == 'reanalysis':
        ds = xr.open_dataset('../reanalysis/CTRL/CRUJRA/'+var+
                             '_CRUJRA_1901-2018_monthly_australia.nc')
    elif method == 'original':
        ds = xr.open_dataset('../CMIP6/CTRL/'+model+'/'+var+'_'+model+
                             '_SSP245_r1i1p1f1_1850_2100_monthly.nc')

    ds_sel = ds.sel(time=slice('1901-01-01', '2018-12-31'))
    return(ds_sel[var])

def error_metric(metric, method, model, var):
    CRUJRA = readin('reanalysis','',var)
    MODEL = readin(method,model,var)
    CRUJRA['time'] = pd.date_range(start='1/1/1901', freq='M',
                                   periods=CRUJRA.time.size)
    MODEL['time'] = pd.date_range(start='1/1/1901', freq='M',
                                  periods=MODEL.time.size)

    if metric == 'RMSE':
        ds_ERROR_map = xs.rmse(CRUJRA, MODEL, dim='time')
        ds_ERROR = ds_ERROR_map.mean().values.tolist()
    elif metric == 'NME':
        ds_diff = abs(CRUJRA - MODEL)
        ds_anomaly = abs(CRUJRA - \
                         CRUJRA.sel(time=slice('1989-01-01',
                                               '2018-01-01')).mean(dim='time'))
        ds_ERROR_map = ds_diff.sum(dim='time')/ds_anomaly.sum(dim='time')
        ds_ERROR = ds_ERROR_map.mean().values.tolist()
    elif metric == 'MBE':
        ds_diff = MODEL - CRUJRA
        ds_ERROR_map = ds_diff.sum(dim='time')/len(ds_diff['time'])
        ds_ERROR = ds_ERROR_map.mean().values.tolist()
    elif metric == 'SD_diff':
        ds_ERROR_map = abs(1-(MODEL.std(dim='time')/CRUJRA.std(dim='time')))
        ds_ERROR = ds_ERROR_map.mean().values.tolist()
    elif metric == 'Corr':
        ds_ERROR_map = 1 - xs.pearson_r(CRUJRA, MODEL, dim='time')
        ds_ERROR = ds_ERROR_map.mean().values.tolist()
    elif metric == '5_PCTL':
        ds_ERROR_map = abs(MODEL.quantile(0.05)-CRUJRA.quantile(0.05))
        ds_ERROR = ds_ERROR_map.mean().values.tolist()
    elif metric == '95_PCTL':
        ds_ERROR_map = abs(MODEL.quantile(0.95)-CRUJRA.quantile(0.95))
        ds_ERROR = ds_ERROR_map.mean().values.tolist()
    elif metric == 'Skewness':
        df_CRUJRA = CRUJRA.to_dataframe().reset_index().drop(columns=['time', 'lat', 'lon'])
        df_MODEL = MODEL.to_dataframe().reset_index().drop(columns=['time', 'lat', 'lon'])
        ds_ERROR = abs(1-(skew(df_MODEL[var],nan_policy='omit')/\
                          skew(df_CRUJRA[var],nan_policy='omit')))
    elif metric == 'Kurtosis':
        df_CRUJRA = CRUJRA.to_dataframe().reset_index().drop(columns=['time', 'lat', 'lon'])
        df_MODEL = MODEL.to_dataframe().reset_index().drop(columns=['time', 'lat', 'lon'])
        ds_ERROR = abs(1-(kurtosis(df_MODEL[var],nan_policy='omit')/ \
                          kurtosis(df_CRUJRA[var],nan_policy='omit')))

    return(ds_ERROR)

df_ERROR = pd.DataFrame(columns=['Model','RMSE', 'NME', 'MBE', 'SD_diff',
                                 'Corr', '5_PCTL', '95_PCTL', 'Skewness',
                                 'Kurtosis'])

metrics = ['RMSE', 'NME', 'MBE', 'SD_diff', 'Corr', '5_PCTL', '95_PCTL',
           'Skewness', 'Kurtosis']
bc_methods = ['SCALING', 'MVA', 'CDFt', 'QM', 'MRec']

model_names = ['CanESM5', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'EC-Earth3',
               'EC-Earth3-Veg', 'GFDL-CM4', 'INM-CM4-8', 'INM-CM5-0',
               'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR',
               'MPI-ESM1-2-LR', 'MRI-ESM2-0','NorESM2-LM', 'NorESM2-MM']

n_rows = np.arange(0,len(model_names)+1)

for i,mn in zip(n_rows, model_names):
    print(mn)
    error_list = [mn]
    for mt in metrics:
        print(mt)
        val = error_metric(mt, 'original', mn, 'temp')
        error_list.append(val)

    df_ERROR.loc[i] = error_list

df_ERROR_index = df_ERROR.set_index('Model')
df_ERROR_standard = (df_ERROR_index - df_ERROR_index.min(axis=0))/\
                    (df_ERROR_index.max(axis=0) - df_ERROR_index.min(axis=0))

fig = plt.figure(figsize=(10,10))

fig.subplots_adjust(hspace=0.12)
fig.subplots_adjust(wspace=0.2)
fig.subplots_adjust(right=0.95)
fig.subplots_adjust(left=0.16)
fig.subplots_adjust(bottom=0.05)
fig.subplots_adjust(top=0.97)

ax1 = fig.add_subplot(1,1,1)

ax1 = sns.heatmap(df_ERROR_standard, annot=df_ERROR.set_index('Model'),
                  cmap='YlGnBu', ax=ax1, cbar_kws={'ticks':[0,0.5,1]})
colorbar = ax1.collections[0].colorbar
colorbar.set_ticklabels(['High skill', 'medium skill', 'low skill'])
ax1.set_ylabel('Model', fontsize=14)
print(df_ERROR_index.rank(method='max').mean(axis=1))

plt.subplot_tool()
plt.show()
# plt.savefig('annual_prec_independent_kendall.pdf')
