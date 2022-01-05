import src.data_processing_module.sampling as sampler
import pandas as pd
from src.data_base_module.data_blocks import TickDataColumns, TickDataFrame, Date, IntraDayPeriod, TimeBarDataFrame, \
    BarDataColumns
import numpy as np

# -------- test case data --------
test_date = Date(day=12, month=5, year=2021)

# ------- bar 1 VWAP = 25.0-------
tick1 = {TickDataColumns.TIMESTAMP_NANO.value: 0, TickDataColumns.LAST_PRICE.value: 10,
         TickDataColumns.LAST_QUANTITY.value: 1}
tick2 = {TickDataColumns.TIMESTAMP_NANO.value: 2 * 1E9, TickDataColumns.LAST_PRICE.value: 20,
         TickDataColumns.LAST_QUANTITY.value: 1}
tick3 = {TickDataColumns.TIMESTAMP_NANO.value: 7 * 1E9, TickDataColumns.LAST_PRICE.value: 30,
         TickDataColumns.LAST_QUANTITY.value: 1}
tick4 = {TickDataColumns.TIMESTAMP_NANO.value: 14 * 1E9, TickDataColumns.LAST_PRICE.value: 40,
         TickDataColumns.LAST_QUANTITY.value: 1}

# ------ bar 2 (empty) all zeros -------
# ------ bar 3 VWAP = 3.9375 -------
tick5 = {TickDataColumns.TIMESTAMP_NANO.value: 35 * 1E9, TickDataColumns.LAST_PRICE.value: 5,
         TickDataColumns.LAST_QUANTITY.value: 5}
tick6 = {TickDataColumns.TIMESTAMP_NANO.value: 35 * 1E9 + 1, TickDataColumns.LAST_PRICE.value: 4,
         TickDataColumns.LAST_QUANTITY.value: 7}
tick7 = {TickDataColumns.TIMESTAMP_NANO.value: 35 * 1E9 + 2, TickDataColumns.LAST_PRICE.value: 3,
         TickDataColumns.LAST_QUANTITY.value: 2}
tick8 = {TickDataColumns.TIMESTAMP_NANO.value: 35 * 1E9 + 3, TickDataColumns.LAST_PRICE.value: 2,
         TickDataColumns.LAST_QUANTITY.value: 2}

# ----- bar 4 VWAP = 19 ------
tick9 = {TickDataColumns.TIMESTAMP_NANO.value: 45 * 1E9, TickDataColumns.LAST_PRICE.value: 19,
         TickDataColumns.LAST_QUANTITY.value: 15}

# ----- bar 5 VWAP = 81 -------
tick10 = {TickDataColumns.TIMESTAMP_NANO.value: 75 * 1E9 - 1, TickDataColumns.LAST_PRICE.value: 81,
          TickDataColumns.LAST_QUANTITY.value: 25}

# ----- bar 6 VWAP = 109 -------
tick11 = {TickDataColumns.TIMESTAMP_NANO.value: 75 * 1E9 + 1, TickDataColumns.LAST_PRICE.value: 109,
          TickDataColumns.LAST_QUANTITY.value: 79}

# ----- bar 7 VWAP = -------
tick12 = {TickDataColumns.TIMESTAMP_NANO.value: 90 * 1E9 + 91286, TickDataColumns.LAST_PRICE.value: 205,
          TickDataColumns.LAST_QUANTITY.value: 40}
tick13 = {TickDataColumns.TIMESTAMP_NANO.value: 90 * 1E9 + 1348722, TickDataColumns.LAST_PRICE.value: 210,
          TickDataColumns.LAST_QUANTITY.value: 30}
tick14 = {TickDataColumns.TIMESTAMP_NANO.value: 90 * 1E9 + 14257883, TickDataColumns.LAST_PRICE.value: 215,
          TickDataColumns.LAST_QUANTITY.value: 30}

# --------- correct answer --------
bar1 = {BarDataColumns.TIMESTAMP.value: 0, BarDataColumns.OPEN.value: 10.0, BarDataColumns.CLOSE.value: 40.0,
        BarDataColumns.HIGH.value: 40.0, BarDataColumns.LOW.value: 10.0, BarDataColumns.VWAP.value: 25.5,
        BarDataColumns.VOLUME.value: 4}
bar2 = {BarDataColumns.TIMESTAMP.value: 0, BarDataColumns.OPEN.value: 0.0, BarDataColumns.CLOSE.value: 0.0,
        BarDataColumns.HIGH.value: 0.0, BarDataColumns.LOW.value: 0.0, BarDataColumns.VWAP.value: 0.0,
        BarDataColumns.VOLUME.value: 0}
bar3 = {BarDataColumns.TIMESTAMP.value: 35 * 1E9, BarDataColumns.OPEN.value: 5.0, BarDataColumns.CLOSE.value: 2.0,
        BarDataColumns.HIGH.value: 5.0, BarDataColumns.LOW.value: 2.0, BarDataColumns.VWAP.value: 3.9375,
        BarDataColumns.VOLUME.value: 16}
bar4 = {BarDataColumns.TIMESTAMP.value: 45 * 1E9, BarDataColumns.OPEN.value: 19.0, BarDataColumns.CLOSE.value: 19.0,
        BarDataColumns.HIGH.value: 19.0, BarDataColumns.LOW.value: 19.0, BarDataColumns.VWAP.value: 19.0,
        BarDataColumns.VOLUME.value: 15}
bar5 = {BarDataColumns.TIMESTAMP.value: 75 * 1E9 - 1, BarDataColumns.OPEN.value: 81.0, BarDataColumns.CLOSE.value: 81.0,
        BarDataColumns.HIGH.value: 81.0, BarDataColumns.LOW.value: 81.0, BarDataColumns.VWAP.value: 81.0,
        BarDataColumns.VOLUME.value: 25}
bar6 = {BarDataColumns.TIMESTAMP.value: 75 * 1E9 + 1, BarDataColumns.OPEN.value: 109.0,
        BarDataColumns.CLOSE.value: 109.0,
        BarDataColumns.HIGH.value: 109.0, BarDataColumns.LOW.value: 109.0, BarDataColumns.VWAP.value: 109.0,
        BarDataColumns.VOLUME.value: 79}
bar7 = {BarDataColumns.TIMESTAMP.value: 90 * 1E9 + 91286, BarDataColumns.OPEN.value: 205.0,
        BarDataColumns.CLOSE.value: 215.0,
        BarDataColumns.HIGH.value: 215.0, BarDataColumns.LOW.value: 205.0, BarDataColumns.VWAP.value: 209.5,
        BarDataColumns.VOLUME.value: 100}

# ------------ set up data frames -----------
# ----- tick data frame -------
raw_tick_df = pd.DataFrame(
    [tick1, tick2, tick3, tick4, tick5, tick6, tick7, tick8, tick9, tick10, tick11, tick12, tick13, tick14])
# ----- fill in the limit order book columns -------
for column in [TickDataColumns.ASK1P.value,
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
tick_df_wrapper = TickDataFrame(tick_df=raw_tick_df, date=test_date, intra_day_period=IntraDayPeriod.WHOLE_DAY,
                                symbol="TEST")
# ------ create answer bar df ------
answer_bar_df = pd.DataFrame([bar1, bar2, bar3, bar4, bar5, bar6, bar7])
# ------ create gibberish column to test redundant column removal of TimeDataFrame Wrapper class ------
answer_bar_df["hachoooo_hachoooo"] = 1
# ------ create answer bar df wrapper -------
answer_bar_df_wrapper = TimeBarDataFrame(bar_df=answer_bar_df, sampling_seconds=15, date=test_date,
                                         intra_day_period=IntraDayPeriod.WHOLE_DAY, symbol="TEST", deep_copy=False)

# ------ sampling -----
result_bar_df_wrapper = sampler.time_sampling(tick_df_wrapper=tick_df_wrapper, sampling_seconds=15)

# ----- print results -------
print(" ------ ANSWER BAR DF -------- ")
print(answer_bar_df_wrapper.get_bar_data_reference())
print(" ------ RESULT BAR DF  -------")
print(result_bar_df_wrapper.get_bar_data_reference())
print("Test case passed : " + str(answer_bar_df.equals(result_bar_df_wrapper.get_bar_data_reference())))

# ----- check time stamp -----
print("checking time stamp")
for ans_ts, result_ts in zip(answer_bar_df_wrapper.get_bar_data_reference()[BarDataColumns.TIMESTAMP.value],
                             result_bar_df_wrapper.get_bar_data_reference()[BarDataColumns.TIMESTAMP.value]):
    print(ans_ts, result_ts)
