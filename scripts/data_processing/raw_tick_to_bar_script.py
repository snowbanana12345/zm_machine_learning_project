from src.data_base_module.data_retrival import instance as db
import src.data_processing_module.sampling as sampling
import src.data_base_module.data_blocks as data

# ------- user inputs ------
sampling_seconds = 60
sampling_ticks = 200
sampling_volume = 20
sampling_dollar = 500000
symbol = "NHK17"
date = data.Date(day = 7, month = 3, year = 2017)

# ------- retrieve raw tick data -------
morning_df_wrapper = db.get_clean_tick_data(symbol = "NHK17", date = date, intra_day_period = data.IntraDayPeriod.MORNING)
afternoon_df_wrapper = db.get_clean_tick_data(symbol = "NHK17", date = date, intra_day_period = data.IntraDayPeriod.AFTERNOON)

# ------- time smapling ------
#morning_time_bar_df = sampling.time_sampling(morning_df_wrapper, sampling_seconds = sampling_seconds)
#afternoon_time_bar_df = sampling.time_sampling(afternoon_df_wrapper, sampling_seconds = sampling_seconds)
#db.insert_sampled_time_bar(morning_time_bar_df)
#db.insert_sampled_time_bar(afternoon_time_bar_df)

# ------- tick sampling ------
#morning_tick_bar_df = sampling.tick_sampling(morning_df_wrapper, sampling_ticks = sampling_ticks)
#afternoon_tick_bar_df = sampling.tick_sampling(afternoon_df_wrapper, sampling_ticks = sampling_ticks)
#db.insert_sampled_tick_bar(morning_tick_bar_df)
#db.insert_sampled_tick_bar(afternoon_tick_bar_df)

# ------- volume sampling -------
morning_volume_bar_df = sampling.volume_sampling(morning_df_wrapper, sampling_volume = sampling_volume)
afternoon_volume_bar_df = sampling.volume_sampling(afternoon_df_wrapper, sampling_volume = sampling_volume)
db.insert_sampled_volume_bar(morning_volume_bar_df)
db.insert_sampled_volume_bar(afternoon_volume_bar_df)

# ------- dollar sampling --------
#morning_dollar_bar_df = sampling.dollar_sampling(morning_df_wrapper, sampling_dollar = sampling_dollar)
#afternoon_dollar_bar_df = sampling.dollar_sampling(afternoon_df_wrapper, sampling_dollar = sampling_dollar)
#db.insert_sampled_dollar_bar(morning_dollar_bar_df)
#db.insert_sampled_dollar_bar(afternoon_dollar_bar_df)