import pandas as pd
from lib import DataInterpolate

# import data
dt = pd.read_csv("AQ_NO2-19800414-20220420.csv", header=4)
dt = dt.values
# station number
index = 2

DI = DataInterpolate(dt, index)
# years can be specified.
# as for NO2, the maximum can be from 1984 to 2021
DI.generate_dataset(48, [2013, 2021])
# DI.cal_missing_length()
# DI.cal_integrity()
DI.period_factor(index=[2])
DI.interpolate('interpolate.interp1d', kind='linear')
print('Linear MSE', DI.cal_mse(DI.validate_data_linear, DI.validate_data_true))
print('Period factor MSE', DI.cal_mse(DI.validate_data_period, DI.validate_data_true))
DI.plot()
