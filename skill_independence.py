import numpy as np
import xarray as xr
import pandas as pd
from numpy.linalg import multi_dot
import os, psutil
from datetime import datetime

startTime = datetime.now()

def readin(var,model,type):
    if model == 'CRUJRA':
        var = ('../../CRUJRA/'+var+'/crujra.v2.0.'+var+'.std-ordering.nc')
    else:
        var = ('../../../australia_climate/'+var+'/'+var+'_'+model+
                   '_SSP245_r1i1p1f1_K_1850_2100.nc')

    ds = xr.open_dataset(var)
    
    ### Select time period to benchmark to 
    if type == 'original':
        ds = ds.sel(time = slice('1989-01-01','2005-12-31'))
        ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    
    ### Select time period for ensemble mean
    elif type == 'toweight':
        ds = ds.sel(time = slice('1851-01-01','2100-12-31'))
        ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))

    lats = ds.lat.values
    lons = ds.lon.values
    time = ds.time.values

    ds_dict = {}
    ds_dict['time'] = ds.time.values
    ds_dict['lon'] =  ds.lon.values
    ds_dict['lat'] =  ds.lat.values

    if model == 'CRUJRA':
        array = ds['prec'].values
        print(array)
    else:
        array_prel = ds['prec'].values
        array = array_prel*86400

    if type == 'toweight':
        if model == 'CanESM5':
            return(ds_dict, time, lats, lons, array)
            print(array)
        else:
            return(array)
    else:
        return(array)

models = ['CanESM5', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg',
          'GFDL-CM4', 'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR',
          'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
          'MRI-ESM2-0', 'NorESM2-LM', 'NorESM2-MM']

can_original = readin('prec','CanESM5','original')
cesm_original = readin('prec','CESM2-WACCM','original')
cmcc_original = readin('prec','CMCC-CM2-SR5','original')
ec_original = readin('prec','EC-Earth3','original')
ecveg_original = readin('prec','EC-Earth3-Veg','original')
gfdl_original = readin('prec','GFDL-CM4','original')
inm4_original = readin('prec','INM-CM4-8','original')
inm5_original = readin('prec','INM-CM5-0','original')
ipsl_original = readin('prec','IPSL-CM6A-LR','original')
kiost_original = readin('prec','KIOST-ESM','original')
miroc_original = readin('prec','MIROC6','original')
mpihr_original = readin('prec','MPI-ESM1-2-HR','original')
mpilr_original = readin('prec','MPI-ESM1-2-LR','original')
mri_original = readin('prec','MRI-ESM2-0','original')
norlm_original = readin('prec','NorESM2-LM','original')
normm_original = readin('prec','NorESM2-MM','original')

model_original_data = [can_original, cesm_original, cmcc_original, ec_original,
                       ecveg_original, gfdl_original, inm4_original,
                       inm5_original, ipsl_original, kiost_original,
                       miroc_original, mpihr_original, mpilr_original,
                       mri_original, norlm_original, normm_original]

hybrid_dict, time, lats, lons, can_toweight = readin('prec','CanESM5','toweight')
cesm_toweight = readin('prec','CESM2-WACCM','toweight')
cmcc_toweight = readin('prec','CMCC-CM2-SR5','toweight')
ec_toweight = readin('prec','EC-Earth3','toweight')
ecveg_toweight = readin('prec','EC-Earth3-Veg','toweight')
gfdl_toweight = readin('prec','GFDL-CM4','toweight')
inm4_toweight = readin('prec','INM-CM4-8','toweight')
inm5_toweight = readin('prec','INM-CM5-0','toweight')
ipsl_toweight = readin('prec','IPSL-CM6A-LR','toweight')
kiost_toweight = readin('prec','KIOST-ESM','toweight')
miroc_toweight = readin('prec','MIROC6','toweight')
mpihr_toweight = readin('prec','MPI-ESM1-2-HR','toweight')
mpilr_toweight = readin('prec','MPI-ESM1-2-LR','toweight')
mri_toweight = readin('prec','MRI-ESM2-0','toweight')
norlm_toweight = readin('prec','NorESM2-LM','toweight')
normm_toweight = readin('prec','NorESM2-MM','toweight')

model_toweight_data = [can_toweight, cesm_toweight, cmcc_toweight, ec_toweight,
                       ecveg_toweight, gfdl_toweight, inm4_toweight,
                       inm5_toweight, ipsl_toweight, kiost_toweight,
                       miroc_toweight, mpihr_toweight, mpilr_toweight,
                       mri_toweight, norlm_toweight, normm_toweight]

cru = readin('prec','CRUJRA','original')
hybrid_matrix = np.zeros([len(time),
                          len(lats),
                          len(lons)])

for i,lat in enumerate(lats):
    for j,lon in enumerate(lons):
            obs = pd.DataFrame()
            obs['CRUJRA']=cru[:,i,j]

            model = pd.DataFrame()
            model_toweight = pd.DataFrame()

            for mn,m  in zip(models,model_original_data):
                model[mn] = m[:,i,j]

            for mn,m in zip(models,model_toweight_data):
                model_toweight[mn] = m[:,i,j]

            le_df_error = model - obs.values

            bc_term = le_df_error.mean(axis=0, skipna=True)
            model_bc = model.copy()
            model_bc = model_bc - bc_term

            error_df = model_bc - obs.values

            M_cov = error_df.cov()
            model_count = len(M_cov.columns)
            unit_col = np.ones((model_count,1))

            M_cov_inv = np.linalg.inv(M_cov)

            unit_transpose = unit_col.transpose()
            weights = np.matmul(M_cov_inv, unit_col)/multi_dot([unit_transpose,
                                                                M_cov_inv,
                                                                unit_col])

            df_bc = pd.DataFrame(bc_term).transpose()
            df_weights = pd.DataFrame(weights.transpose(),columns = models)

            if i%5==0 and j%5==0:
                print(lat,lon)

            hybrid = np.zeros([1, len(model_toweight)])

            for m in models:
                weighted = df_weights[m].values*(model_toweight[m]-df_bc[m].values)
                hybrid = hybrid + weighted.to_numpy()

            hybrid_matrix[:,i,j] = hybrid

dataset = xr.Dataset({'prec':(('time', 'lat','lon'),
                                hybrid_matrix)},
                     coords={'lat': lats,
                             'lon': lons,
                             'time': hybrid_dict['time']})

dataset['lat'].attrs={'units':'degrees_north',
                      'long_name':'latitude',
                      'standard_name':'latitude',
                      'axis':'Y'}
dataset['lon'].attrs={'units':'degrees_east',
                      'long_name':'longitude',
                      'standard_name':'longitude',
                      'axis':'X'}
dataset['prec'].attrs={'long_name':'Precipitation',
                       'standard_name':'precipitation_amount',
                       'units':'kg m-2'}

out_name = 'prec_1851-2100_weighted_CRUJRA.nc'

dataset.to_netcdf(out_name,
                  encoding={'time':{'dtype':'double'},
                            'lat':{'dtype': 'double'},
                            'lon':{'dtype': 'double'},
                            'prec':{'dtype': 'float32'}
                            }
                  )

process = psutil.Process(os.getpid())
print(process.memory_info().rss/(1024 ** 2))
print(datetime.now() - startTime)
