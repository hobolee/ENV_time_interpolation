import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from lib import search_year, cal_valid_time, DataInterpolate
import statsmodels.api as sm
import statsmodels.tools as smt

dt = pd.read_csv("AQ_NO2-19800414-20220420.csv", header=4)
dt = dt.values
index = 0
DI = DataInterpolate(dt, index)
DI.generate_dataset(48, [2019, 2021])
print(DI.operate_X.shape)
print(DI.operate_T.shape)
# plt.plot(DI.operate_X, DI.operate_data[:, index+1])
# plt.show()
# DI.cal_integrity()
# DI.interpolate('interpolate.interp1d', kind=5)
# DI.plot(mode='interpolation')
# DI.cal_missing_length()
DI.period_factor(index=[0])
DI.interpolate('interpolate.interp1d', kind='linear')
print(DI.cal_mse(DI.validate_data_linear, DI.validate_data_true))
print(DI.cal_mse(DI.validate_data_period, DI.validate_data_true))
print(DI.cal_r2(DI.validate_data_linear, DI.validate_data_true))
print(DI.cal_r2(DI.validate_data_period, DI.validate_data_true))
DI.plot()

