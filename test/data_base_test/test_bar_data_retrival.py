import src.data_base_module.data_blocks as data
from src.data_base_module.data_retrival import instance as db
import matplotlib.pyplot as plt
import src.data_processing_module.data_cleaning as dat_clean

symbol : str = "NHK17"
sampling_seconds : int = 60
sampling_volume : int = 20
sampling_ticks : int = 200
sampling_dollar : int = 500000
date : data.Date = data.Date(day = 10, month = 2, year = 2017)
intra_day_period : data.IntraDayPeriod = data.IntraDayPeriod.AFTERNOON
#intra_day_period : data.IntraDayPeriod = data.IntraDayPeriod.MORNING


time_bar : data.BarDataFrame = db.get_sampled_bar(symbol = symbol, sampling_level = sampling_seconds, date = date
                                                  , intra_day_period = intra_day_period, sampling_type = data.Sampling.TIME)
tick_bar : data.BarDataFrame = db.get_sampled_bar(symbol = symbol, sampling_level = sampling_ticks, date = date
                                                  , intra_day_period = intra_day_period, sampling_type = data.Sampling.TICK)
vol_bar : data.BarDataFrame = db.get_sampled_bar(symbol = symbol, sampling_level = sampling_volume, date = date
                                                 , intra_day_period = intra_day_period, sampling_type = data.Sampling.VOLUME)
dollar_bar : data.BarDataFrame = db.get_sampled_bar(symbol = symbol, sampling_level = sampling_dollar, date = date
                                                    , intra_day_period = intra_day_period, sampling_type = data.Sampling.DOLLAR)

time_bar = dat_clean.interpolate_bar_zero_prices(time_bar)
tick_bar = dat_clean.interpolate_bar_zero_prices(tick_bar)

plt.plot(time_bar.get_column(col_name = data.BarDataColumns.CLOSE), label = "time")
plt.title("time")
plt.show()
plt.clf()
plt.plot(tick_bar.get_column(col_name = data.BarDataColumns.CLOSE), label = "tick")
plt.title("tick")
plt.show()
plt.clf()
plt.plot(vol_bar.get_column(col_name = data.BarDataColumns.CLOSE), label = "vol")
plt.title("volume")
plt.show()
plt.clf()
plt.plot(dollar_bar.get_column(col_name = data.BarDataColumns.CLOSE), label = "dollar")
plt.title("dollar")
plt.show()
plt.clf()
