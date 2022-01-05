import src.data_base_module.data_blocks as data
from src.data_base_module.data_retrival import instance as db

symbol : str = "NHK17"
sampling_volume : int = 20
date : data.Date = data.Date(day = 25, month = 1, year = 2017)
intra_day_period : data.IntraDayPeriod = data.IntraDayPeriod.MORNING
vol_bar : data.BarDataFrame = db.get_sampled_volume_bar(symbol = symbol, sampling_volume = sampling_volume, date = date, intra_day_period = intra_day_period)
print(vol_bar)