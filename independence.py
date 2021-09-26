import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl
from scipy.stats import kendalltau, pearsonr, spearmanr

def kendall_pval(x,y):
    return kendalltau(x,y)[1]

def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]

def spearmanr_pval(x,y):
    return spearmanr(x,y)[1]

def readin(param,model):
    if model == 'CRUJRA':
        data = ('../../CRUJRA/'+param+'/crujra.v2.0.'+param+'.std-ordering.nc')
    else:
        data = ('../../../australia_climate/'+param+'/'+param+'_'+model+
                 '_SSP245_r1i1p1f1_K_1850_2100.nc')

    ds = xr.open_dataset(data)
    ds = ds.sel(time = slice('1989-01-01','2005-12-31'))

    if param == 'prec':
        ds_annual = ds.groupby('time.year').sum('time', skipna=False)
    else:
        ds_annual = ds.groupby('time.year').mean('time', skipna=False)

    array = ds_annual[param].values
    return(array)

df_model = pd.DataFrame()

models = ['CanESM5', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg',
          'GFDL-CM4', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR',
          'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
          'MRI-ESM2-0', 'NorESM2-LM', 'NorESM2-MM']

for m in models:
    data = readin('prec', m)
    df_model[m] = data.flatten()

## Adjust precip unit
df_model = df_model*86400

df_obs = pd.DataFrame()
data_obs = readin('prec', 'CRUJRA')
df_obs['obs'] = data_obs.flatten()

df_bias = df_model - df_obs.values

### Different corr coefficients
df_corr = df_bias.corr(method='kendall')

### Arbitrary choice really
df_filter = df_corr.mask(df_corr > 0.3)
df_filter_new = df_filter.mask(df_filter < -0.3)

fig = plt.figure(figsize=(9,8))
plt.subplot(111)
plt.subplots_adjust(bottom=0.18, left=0.15, right=0.85, top=0.95)

levels = np.arange(0,1.1,0.1)

cmap = plt.cm.viridis_r
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = mpl.colors.LinearSegmentedColormap.from_list('mcm', cmaplist, cmap.N)
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

cm = plt.pcolormesh(df_corr, cmap=cmap, norm=norm)
plt.pcolor(df_filter_new, hatch='.', alpha=0.)
plt.yticks(np.arange(0.5, len(df_filter.index), 1), df_filter.index)
plt.xticks(np.arange(0.5, len(df_filter.columns), 1), df_filter.columns,
           rotation='vertical')

cax = plt.axes([0.88, 0.18, 0.04, 0.77])
cbar = plt.colorbar(cm,cax=cax)

cbar.set_ticks(levels)
# cbar.set_ticklabels([mn,md,mx])

# plt.show()
plt.savefig('annual_prec_independent_kendall.pdf')
