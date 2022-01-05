import src.data_processing_module.sampling as sampler
import pandas as pd
from src.data_base_module.data_blocks import TickDataColumns, TickDataFrame, Date, IntraDayPeriod, TickBarDataFrame, BarDataColumns
import numpy as np

# -------- test case , sample into bars of 4 ticks --------
test_date = Date(day=17, month=7, year=2012)

# -------- bar 1 VWAP = 29.5------
tick1 = {TickDataColumns.TIMESTAMP_NANO.value : 101, TickDataColumns.LAST_PRICE.value : 20, TickDataColumns.LAST_QUANTITY.value : 2}
tick2 = {TickDataColumns.TIMESTAMP_NANO.value : 256, TickDataColumns.LAST_PRICE.value : 30, TickDataColumns.LAST_QUANTITY.value : 3}
tick3 = {TickDataColumns.TIMESTAMP_NANO.value : 312, TickDataColumns.LAST_PRICE.value : 25, TickDataColumns.LAST_QUANTITY.value : 1}
tick4 = {TickDataColumns.TIMESTAMP_NANO.value : 498, TickDataColumns.LAST_PRICE.value : 35, TickDataColumns.LAST_QUANTITY.value : 4}

# ------ bar 2 (empty) VWAP = 0 -------
tick5 = {TickDataColumns.TIMESTAMP_NANO.value : 685, TickDataColumns.LAST_PRICE.value : 0, TickDataColumns.LAST_QUANTITY.value : 0}
tick6 = {TickDataColumns.TIMESTAMP_NANO.value : 795, TickDataColumns.LAST_PRICE.value : 0, TickDataColumns.LAST_QUANTITY.value : 0}
tick7 = {TickDataColumns.TIMESTAMP_NANO.value : 982, TickDataColumns.LAST_PRICE.value : 0, TickDataColumns.LAST_QUANTITY.value : 0}
tick8 = {TickDataColumns.TIMESTAMP_NANO.value : 1109, TickDataColumns.LAST_PRICE.value : 0, TickDataColumns.LAST_QUANTITY.value : 0}

# ------ bar 3 (mixture of empty ticks) VWAP = 45 -----
tick9 = {TickDataColumns.TIMESTAMP_NANO.value : 1203, TickDataColumns.LAST_PRICE.value : 0, TickDataColumns.LAST_QUANTITY.value : 0}
tick10 = {TickDataColumns.TIMESTAMP_NANO.value : 1376, TickDataColumns.LAST_PRICE.value : 35, TickDataColumns.LAST_QUANTITY.value : 2}
tick11 = {TickDataColumns.TIMESTAMP_NANO.value : 1416, TickDataColumns.LAST_PRICE.value : 50, TickDataColumns.LAST_QUANTITY.value : 4}
tick12 = {TickDataColumns.TIMESTAMP_NANO.value : 1531, TickDataColumns.LAST_PRICE.value : 0, TickDataColumns.LAST_QUANTITY.value : 0}

# ----- bar 4 (2 ticks) left over ticks VWAP = 45 -------
tick13 = {TickDataColumns.TIMESTAMP_NANO.value : 1785, TickDataColumns.LAST_PRICE.value : 35, TickDataColumns.LAST_QUANTITY.value : 2}
tick14 = {TickDataColumns.TIMESTAMP_NANO.value : 1982, TickDataColumns.LAST_PRICE.value : 50, TickDataColumns.LAST_QUANTITY.value : 4}

# ------ correct answer ---------
bar1 = {BarDataColumns.TIMESTAMP.value : 101, BarDataColumns.OPEN.value : 20.0, BarDataColumns.CLOSE.value : 35.0,
        BarDataColumns.HIGH.value : 35.0, BarDataColumns.LOW.value : 20.0, BarDataColumns.VWAP.value : 29.5, BarDataColumns.VOLUME.value : 10}
bar2 = {BarDataColumns.TIMESTAMP.value : 685, BarDataColumns.OPEN.value : 0, BarDataColumns.CLOSE.value : 0,
        BarDataColumns.HIGH.value : 0, BarDataColumns.LOW.value : 0, BarDataColumns.VWAP.value : 0,  BarDataColumns.VOLUME.value : 0}
bar3 = {BarDataColumns.TIMESTAMP.value : 1203, BarDataColumns.OPEN.value : 35, BarDataColumns.CLOSE.value : 50,
        BarDataColumns.HIGH.value : 50, BarDataColumns.LOW.value : 35, BarDataColumns.VWAP.value : 45,  BarDataColumns.VOLUME.value : 6}
bar4 = {BarDataColumns.TIMESTAMP.value : 1785, BarDataColumns.OPEN.value : 35, BarDataColumns.CLOSE.value : 50,
        BarDataColumns.HIGH.value : 50, BarDataColumns.LOW.value : 35, BarDataColumns.VWAP.value : 45,  BarDataColumns.VOLUME.value : 6}

# ------------ set up data frames -----------
# ----- tick data frame -------
raw_tick_df = pd.DataFrame([tick1, tick2, tick3, tick4, tick5, tick6, tick7, tick8, tick9, tick10, tick11, tick12, tick13, tick14])
# ----- fill in the limit order book columns -------
for column in [ TickDataColumns.ASK1P.value,
                TickDataColumns.ASK2P.value,
                TickDataColumns.ASK3P.value,
                TickDataColumns.ASK4P.value,
                TickDataColumns.ASK5P.value,
                TickDataColumns.ASK1Q.value,
                TickDataColumns.ASK2Q.value,
                TickDataColumns.ASK3Q.value,
                TickDataColumns.ASK4Q.value,
                TickDataColumns.ASK5Q.value,
                TickDataColumns.BID1P.value,
                TickDataColumns.BID2P.value,
                TickDataColumns.BID3P.value,
                TickDataColumns.BID4P.value,
                TickDataColumns.BID5P.value,
                TickDataColumns.BID1Q.value,
                TickDataColumns.BID2Q.value,
                TickDataColumns.BID3Q.value,
                TickDataColumns.BID4Q.value,
                TickDataColumns.BID5Q.value]:
    raw_tick_df[column] = np.zeros(14)
# ------ fill in the key symbol column ------
raw_tick_df[TickDataColumns.KEY_SYMBOL.value] = "TEST"
# ------ create gibberish column to test redundant column removal of TickDataFrame Wrapper class -----
raw_tick_df["alalabaarababababa_bubrabubra"] = 0
# ------ create tick data frame wrapper object -------
tick_df_wrapper = TickDataFrame(tick_df = raw_tick_df, date = test_date, intra_day_period = IntraDayPeriod.WHOLE_DAY, symbol = "TEST")
# ------ create answer bar df ------
answer_bar_df = pd.DataFrame([bar1, bar2, bar3, bar4])
# ------ create gibberish column to test redundant column removal of VolumeDataFrame Wrapper class ------
answer_bar_df["hachoooo_hachoooo"] = 1
# ------ create answer bar df wrapper -------
answer_bar_df_wrapper = TickBarDataFrame(bar_df = answer_bar_df, sampling_ticks = 4, date = test_date, intra_day_period = IntraDayPeriod.WHOLE_DAY, symbol = "TEST", deep_copy = False)

# ------ sampling ------
result_bar_df_wrapper = sampler.tick_sampling(tick_df_wrapper, sampling_ticks = 4)


# ----- print results -------
print(" ------ ANSWER BAR DF -------- ")
print(answer_bar_df_wrapper.get_bar_data_reference())
print(" ------ RESULT BAR DF  -------")
print(result_bar_df_wrapper.get_bar_data_reference())
print("Test case passed : " + str(answer_bar_df.equals(result_bar_df_wrapper.get_bar_data_reference())))