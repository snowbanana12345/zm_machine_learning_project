import src.data_processing_module.sampling as sampler
import pandas as pd
from src.data_base_module.data_blocks import TickDataColumns, TickDataFrame, Date, IntraDayPeriod, DollarBarDataFrame, BarDataColumns
import numpy as np

# -------- test case , sample into bars of dollar 100 --------
test_date = Date(day=17, month=7, year=2012)

# -------- bar 1 volume = 10 (no over flow) -------
tick1 = {TickDataColumns.TIMESTAMP_NANO.value : 175, TickDataColumns.LAST_PRICE.value : 10, TickDataColumns.LAST_QUANTITY.value : 2}
tick2 = {TickDataColumns.TIMESTAMP_NANO.value : 323, TickDataColumns.LAST_PRICE.value : 15, TickDataColumns.LAST_QUANTITY.value : 1}
tick3 = {TickDataColumns.TIMESTAMP_NANO.value : 477, TickDataColumns.LAST_PRICE.value : 20, TickDataColumns.LAST_QUANTITY.value : 2}
tick4 = {TickDataColumns.TIMESTAMP_NANO.value : 596, TickDataColumns.LAST_PRICE.value : 5, TickDataColumns.LAST_QUANTITY.value : 5}

# ------- bar 2 volume = 6.6 (over flow by 24) ------
tick5 = {TickDataColumns.TIMESTAMP_NANO.value : 700, TickDataColumns.LAST_PRICE.value : 13, TickDataColumns.LAST_QUANTITY.value : 3}
tick6 = {TickDataColumns.TIMESTAMP_NANO.value : 775, TickDataColumns.LAST_PRICE.value : 11, TickDataColumns.LAST_QUANTITY.value : 1}
tick7 = {TickDataColumns.TIMESTAMP_NANO.value : 857, TickDataColumns.LAST_PRICE.value : 22, TickDataColumns.LAST_QUANTITY.value : 2}
tick8 = {TickDataColumns.TIMESTAMP_NANO.value : 1010, TickDataColumns.LAST_PRICE.value : 10, TickDataColumns.LAST_QUANTITY.value : 3}

# ------ bar 3 (carry over 24, 2.4 volume from previous bar ) volume = 5.4 -------
tick9 = {TickDataColumns.TIMESTAMP_NANO.value : 1331, TickDataColumns.LAST_PRICE.value : 29, TickDataColumns.LAST_QUANTITY.value : 2}
tick10 = {TickDataColumns.TIMESTAMP_NANO.value : 1511, TickDataColumns.LAST_PRICE.value : 18, TickDataColumns.LAST_QUANTITY.value : 1}

# ------ bar 4 (mixture of empty ticks) ----------
tick11 = {TickDataColumns.TIMESTAMP_NANO.value : 1545, TickDataColumns.LAST_PRICE.value : 10, TickDataColumns.LAST_QUANTITY.value : 2}
tick12 = {TickDataColumns.TIMESTAMP_NANO.value : 1687, TickDataColumns.LAST_PRICE.value : 0, TickDataColumns.LAST_QUANTITY.value : 0}
tick13 = {TickDataColumns.TIMESTAMP_NANO.value : 1778, TickDataColumns.LAST_PRICE.value : 15, TickDataColumns.LAST_QUANTITY.value : 1}
tick14 = {TickDataColumns.TIMESTAMP_NANO.value : 1928, TickDataColumns.LAST_PRICE.value : 0, TickDataColumns.LAST_QUANTITY.value : 0}
tick15 = {TickDataColumns.TIMESTAMP_NANO.value : 1999, TickDataColumns.LAST_PRICE.value : 20, TickDataColumns.LAST_QUANTITY.value : 2}
tick16 = {TickDataColumns.TIMESTAMP_NANO.value : 2012, TickDataColumns.LAST_PRICE.value : 0, TickDataColumns.LAST_QUANTITY.value : 0}
tick17 = {TickDataColumns.TIMESTAMP_NANO.value : 2304, TickDataColumns.LAST_PRICE.value : 5, TickDataColumns.LAST_QUANTITY.value : 5}

# ------ left over ticks --------
tick18 = {TickDataColumns.TIMESTAMP_NANO.value : 2399, TickDataColumns.LAST_PRICE.value : 5, TickDataColumns.LAST_QUANTITY.value : 5}
tick19 = {TickDataColumns.TIMESTAMP_NANO.value : 2454, TickDataColumns.LAST_PRICE.value : 5, TickDataColumns.LAST_QUANTITY.value : 5}

# --------- correct answer --------
bar1 = {BarDataColumns.TIMESTAMP.value : 175, BarDataColumns.OPEN.value : 10, BarDataColumns.CLOSE.value : 5,
        BarDataColumns.HIGH.value : 20, BarDataColumns.LOW.value : 5, BarDataColumns.VWAP.value : 100 / 10, BarDataColumns.VOLUME.value : 10}
bar2 = {BarDataColumns.TIMESTAMP.value : 700, BarDataColumns.OPEN.value : 13, BarDataColumns.CLOSE.value : 10,
        BarDataColumns.HIGH.value : 22, BarDataColumns.LOW.value : 10, BarDataColumns.VWAP.value : 100 / 6.6, BarDataColumns.VOLUME.value : 6.6}
bar3 = {BarDataColumns.TIMESTAMP.value : 1010, BarDataColumns.OPEN.value : 10, BarDataColumns.CLOSE.value : 18,
        BarDataColumns.HIGH.value : 29, BarDataColumns.LOW.value : 10, BarDataColumns.VWAP.value : 100 / 5.4, BarDataColumns.VOLUME.value : 5.4}
bar4 = {BarDataColumns.TIMESTAMP.value : 1545, BarDataColumns.OPEN.value : 10, BarDataColumns.CLOSE.value : 5,
        BarDataColumns.HIGH.value : 20, BarDataColumns.LOW.value : 5, BarDataColumns.VWAP.value : 100 / 10, BarDataColumns.VOLUME.value : 10}

# ------------ set up data frames -----------
# ----- tick data frame -------
raw_tick_df = pd.DataFrame([tick1, tick2, tick3, tick4, tick5, tick6, tick7, tick8, tick9, tick10, tick11, tick12, tick13, tick14, tick15, tick16, tick17, tick18, tick19])
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
    raw_tick_df[column] = np.zeros(19)
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
answer_bar_df_wrapper = DollarBarDataFrame(bar_df = answer_bar_df, sampling_dollar = 100, date = test_date, intra_day_period = IntraDayPeriod.WHOLE_DAY, symbol = "TEST", deep_copy = False)

# ------ sampling ------
result_bar_df_wrapper = sampler.dollar_sampling(tick_df_wrapper, sampling_dollar = 100)
result_bar_df = result_bar_df_wrapper.get_bar_data_reference()

# ----- print results -------
print(" ------ ANSWER BAR DF -------- ")
print(answer_bar_df_wrapper.get_bar_data_reference())
print(" ------ RESULT BAR DF  -------")
print(result_bar_df_wrapper.get_bar_data_reference())
print("Test case passed : " + str(answer_bar_df.equals(result_bar_df)))