from src.data_base_module.data_retrival import instance as db
import src.data_base_module.data_blocks as data
import src.plotting_module.bar_data_plotter as bar_plotter

# ------- user inputs ------
sampling_seconds = 60
sampling_ticks = 200
sampling_volume = 20
sampling_dollar = 500000
symbol = "NHK17"
date = data.Date(day = 7, month = 3, year = 2017)


# ------ retrieve bar data -------
morning_time_bar_wrapper : data.TimeBarDataFrame = db.get_sampled_time_bar(symbol = symbol, date = date, intra_day_period = data.IntraDayPeriod.MORNING, sampling_seconds = sampling_seconds)
morning_tick_bar_wrapper : data.TickBarDataFrame = db.get_sampled_tick_bar(symbol = symbol, date = date, intra_day_period = data.IntraDayPeriod.MORNING, sampling_ticks = sampling_ticks)
morning_volume_bar_wrapper : data.VolumeBarDataFrame = db.get_sampled_volume_bar(symbol = symbol, date = date, intra_day_period = data.IntraDayPeriod.MORNING, sampling_volume = sampling_volume)
morning_dollar_bar_wrapper : data.DollarBarDataFrame = db.get_sampled_dollar_bar(symbol = symbol, date = date, intra_day_period = data.IntraDayPeriod.MORNING, sampling_dollar = sampling_dollar)

afternoon_time_bar_wrapper : data.TimeBarDataFrame = db.get_sampled_time_bar(symbol = symbol, date = date, intra_day_period = data.IntraDayPeriod.AFTERNOON, sampling_seconds = sampling_seconds)
afternoon_tick_bar_wrapper : data.TickBarDataFrame = db.get_sampled_tick_bar(symbol = symbol, date = date, intra_day_period = data.IntraDayPeriod.AFTERNOON, sampling_ticks = sampling_ticks)
afternoon_volume_bar_wrapper : data.VolumeBarDataFrame = db.get_sampled_volume_bar(symbol = symbol, date = date, intra_day_period = data.IntraDayPeriod.AFTERNOON, sampling_volume = sampling_volume)
afternoon_dollar_bar_wrapper : data.DollarBarDataFrame = db.get_sampled_dollar_bar(symbol = symbol, date = date, intra_day_period = data.IntraDayPeriod.AFTERNOON, sampling_dollar = sampling_dollar)

# ------ make bar plots ------
bar_plotter.plot_and_save_time_bar(morning_time_bar_wrapper)
bar_plotter.plot_and_save_time_bar(afternoon_time_bar_wrapper)
bar_plotter.plot_and_save_tick_bar(morning_tick_bar_wrapper)
bar_plotter.plot_and_save_tick_bar(afternoon_tick_bar_wrapper)
bar_plotter.plot_and_save_volume_bar(morning_volume_bar_wrapper)
bar_plotter.plot_and_save_volume_bar(afternoon_volume_bar_wrapper)
bar_plotter.plot_and_save_dollar_bar(morning_dollar_bar_wrapper)
bar_plotter.plot_and_save_dollar_bar(afternoon_dollar_bar_wrapper)