import src.data_base_module.data_blocks as data
from src.data_base_module.data_retrival import instance as db
import matplotlib.pyplot as plt

tick_wrapper = db.get_raw_tick_data("NHK17", data.Date(day = 31, month = 1, year = 2017))
tick_df = tick_wrapper.tick_data
print(tick_df.head())
print(tick_wrapper.tick_info)
plt.plot(tick_df.ask1p)
plt.show()

