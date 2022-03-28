import src.data_base_module.data_blocks as data
from src.data_base_module.data_retrival import instance as db

tick_df_wrapper = db.get_raw_tick_data("NHK17", data.Date(day = 25, month = 1, year = 2017))
tick_df_ref = tick_df_wrapper.get_tick_data()
print(tick_df_ref.head())
print(tick_df_wrapper.date.get_str())
print(tick_df_wrapper.symbol)
print(tick_df_wrapper.intra_day_period)