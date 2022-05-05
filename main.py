import pandas as pd
from lib import DataInterpolate


# import data
dt = pd.read_csv("AQ_NO2-19800414-20220420.csv", header=4, dtype=str)
dt = dt.values
# station number
index = 0

DI = DataInterpolate(dt, index)
DI.cal_missing_length()
# DI.cal_integrity()

# years can be specified.
# as for NO2, the maximum should refer to valid time of different stations
DI.generate_dataset(48, [1998, 2021])
DI.period_factor(index=[0])
DI.interpolate('interpolate.interp1d', kind='linear')
print('Linear RMSE', DI.cal_rmse(DI.validate_data_linear, DI.validate_data_true))
print('Period factor MSE', DI.cal_rmse(DI.validate_data_period, DI.validate_data_true))
DI.plot()
