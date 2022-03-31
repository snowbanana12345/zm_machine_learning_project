import pandas as pd
import numpy as np
import time
import src.data_base_module.data_blocks as data
import src.data_processing_module.data_processing_logger as dp_logger


# --------------------- sampling functions ---------------------------
""" Currently supported sampling are time, tick, volume dollar """
def get_nano_time_to_date_time_func(timezone_shift_hour: int) -> callable:
    def nano_time_to_date_time(timestamp: int) -> pd.Timestamp:
        return pd.Timestamp(timestamp) + pd.to_timedelta(timezone_shift_hour, unit="h")
    return nano_time_to_date_time


def time_sampling(tick_wrapper: data.TickDataFrame, sampling_seconds: int = 60) -> data.BarDataFrame:
    """
    Makes use of the resampling function of pandas data frames to resample tick data into data frames
    Only samples ticks with trades occuring.
    :param tick_df: data frame containing tick data
    :param sampling_seconds : the number of seconds in each time bar
    :return pd.DataFrame
    NOTE : time sampling can cause missing bars, these bars will have all their values set to 0.
    """
    # ------ get underlying reference from the tick data frame wrapper ------
    tick_df: pd.DataFrame = tick_wrapper.tick_data
    # ------ log start of sampling process -------
    dp_logger.log_time_sampling_start(tick_info = tick_wrapper.tick_info,sampling_seconds=sampling_seconds)
    # ------ set the index to be date time --------
    tick_df.index = tick_df[data.TickDataColumns.TIMESTAMP_NANO.value].apply(lambda ts: pd.Timestamp(ts))
    # ------ extract the relevant tick information ---------
    rule = str(sampling_seconds) + "S"
    price_col = tick_df[data.TickDataColumns.LAST_PRICE.value]
    qty_col = tick_df[data.TickDataColumns.LAST_QUANTITY.value]
    price_col = price_col[price_col > 0]
    qty_col = qty_col[qty_col > 0]
    timestamps = tick_df[data.TickDataColumns.TIMESTAMP_NANO.value]
    # ----- use pandas resampling functions ----------
    open_price_series = price_col.resample(rule).first()
    close_price_series = price_col.resample(rule).last()
    high_price_series = price_col.resample(rule).max()
    low_price_series = price_col.resample(rule).min()
    volume_traded_series = qty_col.resample(rule).sum()
    timestamps_series = timestamps.resample(rule).first()
    vol_price_product = (price_col * qty_col).resample(rule).sum()
    volume_weighted_average_price_series = vol_price_product.div(volume_traded_series)
    # ----- organize the newly created columns --------
    new_df = pd.DataFrame({
        data.BarDataColumns.OPEN.value: open_price_series,
        data.BarDataColumns.CLOSE.value: close_price_series,
        data.BarDataColumns.HIGH.value: high_price_series,
        data.BarDataColumns.LOW.value: low_price_series,
        data.BarDataColumns.VOLUME.value: volume_traded_series,
        data.BarDataColumns.VWAP.value: volume_weighted_average_price_series,
        data.BarDataColumns.TIMESTAMP.value: timestamps_series
    })
    new_df.fillna(0, inplace=True)
    # ----- reset the index of the input tick_df to range indexing ------
    tick_df.index = pd.RangeIndex(len(tick_df))
    # ----- log end of sampling ------
    dp_logger.log_time_sampling_end(tick_info = tick_wrapper.tick_info, sampling_seconds=sampling_seconds)
    bar_info = data.BarInfo(symbol = tick_wrapper.tick_info.symbol, date = tick_wrapper.tick_info.date, intra_day_period = tick_wrapper.tick_info.intra_day_period
                            , sampling_level = sampling_seconds, sampling_type = data.Sampling.TIME)
    return data.BarDataFrame(bar_data = new_df, bar_info = bar_info)


def volume_sampling(tick_wrapper: data.TickDataFrame, sampling_volume: int = 50) -> data.BarDataFrame:
    """
    combines tick data into volume bars of size specified by bar_size
    Only samples ticks with trades occuring.
    :param tick_df : data frame containing tick data
    :param sampling_volume : the volume traded size of each bar
    NOTE : the last bit of data that does not form a bar is dropped
    """
    # ------ get underlying reference from the tick data frame wrapper ------
    tick_df : pd.DataFrame = tick_wrapper.tick_data
    # ------ log start of sampling process -------
    dp_logger.log_volume_sampling_start(tick_info = tick_wrapper.tick_info, sampling_volume=sampling_volume)
    # ------ initialize --------
    df = tick_df[tick_df[data.TickDataColumns.LAST_PRICE.value] > 0]
    open_price_series = []
    close_price_series = []
    high_price_series = []
    low_price_series = []
    vwap_series = []
    timestamp_series = []
    # ------------ volume sampling  --------------
    volume_counter = 0
    curr_time_stamps: [int] = []
    curr_prices: [float] = []
    curr_volumes: [float] = []
    for i, tick_row in df.iterrows():
        # ------ form bars when there is sufficient volume --------
        while volume_counter >= sampling_volume:
            # ------- calculate the bar values OCHL --------
            first_time_stamp = curr_time_stamps[0]
            volume_counter = volume_counter - sampling_volume
            open_price = float(curr_prices[0])
            close_price = float(curr_prices[-1])
            high_price = float(max(curr_prices))
            low_price = float(min(curr_prices))
            VWAP = sum([vol * price for vol, price in zip(curr_volumes, curr_prices)]) / sampling_volume
            # ------- store bar values --------
            open_price_series.append(open_price)
            close_price_series.append(close_price)
            high_price_series.append(high_price)
            low_price_series.append(low_price)
            vwap_series.append(VWAP)
            timestamp_series.append(first_time_stamp)
            # ------- check for residue volume and overflow --------
            if volume_counter == 0:
                curr_prices = []
                curr_volumes = []
                curr_time_stamps = []
            else:
                curr_prices = [close_price, ]
                curr_volumes = [min(volume_counter, sampling_volume)]
                curr_time_stamps = curr_time_stamps[-1:]
        # ------- collect ticks to form a new bar ----------
        volume_counter = volume_counter + tick_row[data.TickDataColumns.LAST_QUANTITY.value]
        curr_prices.append(tick_row[data.TickDataColumns.LAST_PRICE.value])
        curr_time_stamps.append(tick_row[data.TickDataColumns.TIMESTAMP_NANO.value])
        if volume_counter >= sampling_volume:
            curr_volumes.append(sampling_volume - volume_counter + tick_row[data.TickDataColumns.LAST_QUANTITY.value])
        else:
            curr_volumes.append(tick_row[data.TickDataColumns.LAST_QUANTITY.value])
    # ------ returns a VolumeBarDataFrame object --------
    new_bar_df = pd.DataFrame({
        data.BarDataColumns.OPEN.value: open_price_series,
        data.BarDataColumns.CLOSE.value: close_price_series,
        data.BarDataColumns.HIGH.value: high_price_series,
        data.BarDataColumns.LOW.value: low_price_series,
        data.BarDataColumns.VWAP.value: vwap_series,
        data.BarDataColumns.TIMESTAMP.value: timestamp_series,
        data.BarDataColumns.VOLUME.value : sampling_volume
    })
    bar_info = data.BarInfo(symbol = tick_wrapper.tick_info.symbol, sampling_level = sampling_volume, date = tick_wrapper.tick_info.date,
                            intra_day_period = tick_wrapper.tick_info.intra_day_period, sampling_type = data.Sampling.VOlUME)
    volume_bar_wrapper =  data.BarDataFrame(bar_data = new_bar_df, bar_info = bar_info)
    # ----- log end of sampling ------
    dp_logger.log_volume_sampling_end(tick_info = tick_wrapper.tick_info, sampling_volume=sampling_volume)
    return volume_bar_wrapper

def tick_sampling(tick_wrapper: data.TickDataFrame, sampling_ticks: int = 20) -> data.BarDataFrame:
    """ Combines tick data into bunches of bar_size number of ticks, 
        only ticks with non zero trade volume is counted
    :param tick_df : data frame containing tick data
    :param sampling_ticks : the number of ticks per bar
    NOTE : left over ticks are combined into the last bar
    NOTE : the timestamp is the time stamp of the first tick in the bar """
    # ------ get underlying reference from the tick data frame wrapper ------
    tick_df = tick_wrapper.tick_data
    # ------ log start of sampling process -------
    dp_logger.log_tick_sampling_start(tick_info = tick_wrapper.tick_info, sampling_ticks=sampling_ticks)
    # --------- get relevant columns --------
    price_col: pd.Series = tick_df[data.TickDataColumns.LAST_PRICE.value]
    qty_col: pd.Series = tick_df[data.TickDataColumns.LAST_QUANTITY.value]
    time_col: pd.Series = tick_df[data.TickDataColumns.TIMESTAMP_NANO.value]
    # --------- reindex to a range index --------
    price_col.index = pd.RangeIndex(0, len(price_col), 1)
    qty_col.index = pd.RangeIndex(0, len(qty_col), 1)
    time_col.index = pd.RangeIndex(0, len(time_col), 1)

    # -------- aggregate functions -------
    def find_open_non_zero(series: pd.Series) -> float:
        series = series[series > 0]
        return series.iloc[0] if len(series) > 0 else 0

    def find_close_non_zero(series: pd.Series) -> float:
        series = series[series > 0]
        return series.iloc[-1] if len(series) > 0 else 0

    def find_high_non_zero(series: pd.Series) -> float:
        series = series[series > 0]
        return max(series) if len(series) > 0 else 0

    def find_low_non_zero(series: pd.Series) -> float:
        series = series[series > 0]
        return min(series) if len(series) > 0 else 0

    # -------- find bar features -------
    open_price_series = price_col.groupby(price_col.index // sampling_ticks).agg(find_open_non_zero)
    close_price_series = price_col.groupby(price_col.index // sampling_ticks).agg(find_close_non_zero)
    high_price_series = price_col.groupby(price_col.index // sampling_ticks).agg(find_high_non_zero)
    low_price_series = price_col.groupby(price_col.index // sampling_ticks).agg(find_low_non_zero)
    vol_price_product = price_col.mul(qty_col).groupby(price_col.index // sampling_ticks).sum()
    volume_traded_series = qty_col.groupby(qty_col.index // sampling_ticks).sum()
    vwap_series = vol_price_product.div(volume_traded_series).fillna(0)
    timestamp_series = time_col.groupby(time_col.index // sampling_ticks).first()

    # ------ returns a tick bar data frame object ------
    new_bar_df = pd.DataFrame({
        data.BarDataColumns.OPEN.value: open_price_series,
        data.BarDataColumns.CLOSE.value: close_price_series,
        data.BarDataColumns.HIGH.value: high_price_series,
        data.BarDataColumns.LOW.value: low_price_series,
        data.BarDataColumns.VOLUME.value: volume_traded_series,
        data.BarDataColumns.VWAP.value: vwap_series,
        data.BarDataColumns.TIMESTAMP.value: timestamp_series
    })
    # ----- create TickBarDataFrame object -----
    bar_info = data.BarInfo(symbol = tick_wrapper.tick_info.symbol, date = tick_wrapper.tick_info.date, sampling_level = sampling_ticks,
                            intra_day_period = tick_wrapper.tick_info.intra_day_period, sampling_type = data.Sampling.TICK)
    bar_wrapper = data.BarDataFrame(bar_data = new_bar_df, bar_info = bar_info)
    # ----- log end of sampling ------
    dp_logger.log_tick_sampling_end(tick_info = tick_wrapper.tick_info, sampling_ticks=sampling_ticks)
    return bar_wrapper

def dollar_sampling(tick_wrapper: data.TickDataFrame, sampling_dollar: int = 1000000) -> data.BarDataFrame:
    """
    combines tick data into dollar bars of size specified by bar_size
    NOTE : the last bit of data that does not form a bar is dropped
    """
    # ------ get underlying reference from the tick data frame wrapper ------
    tick_df = tick_wrapper.tick_data
    # ------ log start of sampling process -------
    dp_logger.log_dollar_sampling_start(tick_info = tick_wrapper.tick_info, sampling_dollar=sampling_dollar)
    # ------ dollar sampling ------
    df = tick_df[tick_df[data.TickDataColumns.LAST_PRICE.value] > 0]
    open_price_series_lst : [float] = []
    close_price_series_lst : [float] = []
    high_price_series : [float] = []
    low_price_series : [float] = []
    volume_traded_series : [int]= []
    timestamp_series : [int] = []
    # ------------ dollar sampling  --------------
    new_bar = True
    dollar_counter = 0
    prev_dollar_counter = 0
    time_stamp = 0
    curr_prices = []
    curr_volumes = []
    for i, row in df.iterrows():
        # ------ form bars when there is sufficient volume --------
        while dollar_counter >= sampling_dollar:
            # ------- calculate the bar values OCHLV --------
            dollar_counter = dollar_counter - sampling_dollar
            open_price = curr_prices[0]
            close_price = curr_prices[-1]
            high_price = max(curr_prices)
            low_price = min(curr_prices)
            last_tick_total_volume = curr_volumes[-1]
            last_tick_fill = sampling_dollar - prev_dollar_counter  # the amount of the dollar bar filled by the last tick
            prev_tick_volumes = sum(curr_volumes[:-1])  # the volume of all ticks except the last volume
            last_tick_volume = last_tick_fill / close_price  # the volume filled by the last tick
            bar_volume = last_tick_volume + prev_tick_volumes  # the total volume in this bar
            # ------- store bar values --------
            open_price_series_lst.append(open_price)
            close_price_series_lst.append(close_price)
            high_price_series.append(high_price)
            low_price_series.append(low_price)
            volume_traded_series.append(bar_volume)
            timestamp_series.append(time_stamp)
            # ------- check for residue dollar and overflow --------
            if dollar_counter >= sampling_dollar:
                #  ----- the tick has so much dollar that it can fill multiple bars ------
                curr_prices = [close_price, ]
                curr_volumes = [last_tick_total_volume - last_tick_volume]
                prev_dollar_counter = 0
            elif dollar_counter == 0:
                curr_prices = []
                curr_volumes = []
                prev_dollar_counter = 0
            else:
                # ------ the is a left over amount < bar_size which is to be carried forward to the next bar ------
                curr_prices = [close_price, ]
                curr_volumes = [last_tick_total_volume - last_tick_volume]
                prev_dollar_counter = dollar_counter
            new_bar = True
        # ------- collect ticks to form a new bar ----------
        if new_bar:
            time_stamp = row[data.TickDataColumns.TIMESTAMP_NANO.value]
            new_bar = False
        dollar_counter = dollar_counter + row[data.TickDataColumns.LAST_QUANTITY.value] * row[
            data.TickDataColumns.LAST_PRICE.value]
        curr_prices.append(row[data.TickDataColumns.LAST_PRICE.value])
        curr_volumes.append(row[data.TickDataColumns.LAST_QUANTITY.value])
        if dollar_counter < sampling_dollar:
            prev_dollar_counter = dollar_counter

    # ----- convert to a bar data frame object ------
    vwap_series = pd.Series([sampling_dollar for _ in range(len(volume_traded_series))]).div(volume_traded_series)
    result_bar_df = pd.DataFrame({
        data.BarDataColumns.OPEN.value: open_price_series_lst,
        data.BarDataColumns.CLOSE.value: close_price_series_lst,
        data.BarDataColumns.HIGH.value: high_price_series,
        data.BarDataColumns.LOW.value: low_price_series,
        data.BarDataColumns.VOLUME.value: volume_traded_series,
        data.BarDataColumns.TIMESTAMP.value: timestamp_series,
        data.BarDataColumns.VWAP.value : vwap_series
    })
    bar_info = data.BarInfo(symbol = tick_wrapper.tick_info.symbol, sampling_level = sampling_dollar, date = tick_wrapper.tick_info.date,
                            intra_day_period = tick_wrapper.tick_info.intra_day_period, sampling_type = data.Sampling.DOLLAR)
    dollar_bar_df_wrapper = data.BarDataFrame(bar_data = result_bar_df, bar_info = bar_info)
    # ----- log end of sampling ------
    dp_logger.log_dollar_sampling_end(tick_info = tick_wrapper.tick_info, sampling_dollar=sampling_dollar)
    return dollar_bar_df_wrapper

# -------- bar sampling with statistics on limit order books ----------
def volume_sampling_limit_book(tick_df, bar_size, last_price_col="lastPrice", last_qty_col="lastQty",
                               time_stamp_col="timestampNano",
                               best_ask_col="best_ask", best_bid_col="best_bid", save=None):
    if len(tick_df) == 0:
        if save:
            new_df = pd.DataFrame({"timestamp": [], "open": [], "close": [], "high": [], "low": [], "VVAP": [],
                                   "open_bid": [], "high_bid": [], "low_bid": [], "close_bid": [], "average_bid": [],
                                   "open_ask": [], "close_ask": [], "high_ask": [], "low_ask": [], "average_ask": [],
                                   "average_bid_ask_spread": []})
            new_df.to_csv(save)
        return tick_df
    start_time = time.time()
    """
    combines tick data into volume bars of size specified by bar_size
    NOTE : the last bit of data that does not form a bar is dropped
    NOTE : 
    """
    df = tick_df[tick_df[last_price_col] > 0]
    new_df = pd.DataFrame({"timestamp": [], "open": [], "close": [], "high": [], "low": [], "VVAP": [],
                           "open_bid": [], "high_bid": [], "low_bid": [], "close_bid": [], "average_bid": [],
                           "open_ask": [], "close_ask": [], "high_ask": [], "low_ask": [], "average_ask": [],
                           "average_bid_ask_spread": [], "std_bid_ask_spread": []})
    new_bar = True
    volume_counter = 0
    time_stamp = 0
    curr_prices = []
    curr_volumes = []
    best_asks = []
    best_bids = []
    bid_ask_spreads = []
    for i, row in df.iterrows():
        while volume_counter >= bar_size:
            # ----- trade statistics ------
            volume_counter = volume_counter - bar_size
            open_price = curr_prices[0]
            close_price = curr_prices[-1]
            high_price = max(curr_prices)
            low_price = min(curr_prices)
            # ----- bid ask statistics ------
            open_bid = best_bids[0]
            close_bid = best_bids[-1]
            high_bid = max(best_bids)
            low_bid = min(best_bids)
            average_bid = np.mean(best_bids)

            open_ask = best_asks[0]
            close_ask = best_asks[-1]
            high_ask = max(best_asks)
            low_ask = min(best_asks)
            average_ask = np.mean(best_asks)

            average_bid_ask_spread = np.mean(bid_ask_spreads)
            std_bid_ask_spread = np.std(bid_ask_spreads)

            VVAP = sum([vol * price for vol, price in zip(curr_volumes, curr_prices)]) / bar_size
            new_bar = {"timestamp": time_stamp, "open": open_price, "close": close_price, "high": high_price,
                       "low": low_price, "VVAP": VVAP,
                       "open_bid": open_bid, "close_bid": close_bid, "high_bid": high_bid, "low_bid": low_bid,
                       "average_bid": average_bid,
                       "open_ask": open_ask, "close_ask": close_ask, "high_ask": high_ask, "low_ask": low_ask,
                       "average_ask": average_ask,
                       "average_bid_ask_spread": average_bid_ask_spread, "std_bid_ask_spread": std_bid_ask_spread
                       }
            new_df = new_df.append(new_bar, ignore_index=True)
            if volume_counter == 0:  # if there is no left over volume, we have hit exactly bar_size amount.
                curr_prices = []
                curr_volumes = []
                best_asks = []
                best_bids = []
                bid_ask_spreads = []
            else:  # there is still some volume left over
                curr_prices = [close_price, ]  # the close price of previous bar becomes the new open price
                curr_volumes = [min(volume_counter, bar_size)]  # if spill over volume exceeds bar_size amount
                best_asks = [close_ask, ]
                best_bids = [close_bid, ]
                bid_ask_spreads = bid_ask_spreads[-1:]
            new_bar = True
        if new_bar:
            time_stamp = row[time_stamp_col]
            new_bar = False
        volume_counter = volume_counter + row[last_qty_col]
        curr_prices.append(row[last_price_col])
        if volume_counter >= bar_size:
            curr_volumes.append(bar_size - volume_counter + row[last_qty_col])
        else:
            curr_volumes.append(row[last_qty_col])
        best_asks.append(row[best_ask_col])
        best_bids.append(row[best_bid_col])
        bid_ask_spreads.append(row[best_ask_col] - row[best_bid_col])

    new_df["volume_time"] = list(range(0, len(new_df) * bar_size, bar_size))
    new_df["timestamp"] = new_df["timestamp"].mask(lambda x: x == 0).interpolate().ffill().bfill()
    if save:
        new_df.to_csv(save, index=False)
    end_time = time.time()
    print("Volume sampling completed : elasped time : " + str(end_time - start_time))
    return new_df


def tick_sampling_limit_book(tick_df, bar_size=20, last_price_col="lastPrice", last_qty_col="lastQty",
                             time_stamp_col="timestampNano",
                             best_ask_col="best_ask", best_bid_col="best_bid", save=None):
    if len(tick_df) == 0:
        if save:
            new_df = pd.DataFrame(
                {"timestamp": [], "open": [], "close": [], "high": [], "low": [], "volume": [], "VVAP": [],
                 "tick_time": [],
                 "open_bid": [], "high_bid": [], "low_bid": [], "close_bid": [], "average_bid": [],
                 "open_ask": [], "close_ask": [], "high_ask": [], "low_ask": [], "average_ask": [],
                 "average_bid_ask_spread": [], "std_bid_ask_spread": []})
            new_df.to_csv(save)
        return tick_df
    start_time = time.time()
    """ Combines tick data into bunches of bar_size number of ticks, 
        only ticks with non zero trade volume is counted 
    NOTE : left over ticks are combined into the last bar """
    price_col = tick_df[last_price_col]
    qty_col = tick_df[last_qty_col]
    time_col = tick_df[time_stamp_col]
    ask_col = tick_df[best_ask_col]
    bid_col = tick_df[best_bid_col]
    # --------- filter out 0 rows --------
    time_col = time_col[qty_col > 0]
    price_col = price_col[price_col > 0]
    ask_col = ask_col[qty_col > 0]
    bid_col = bid_col[qty_col > 0]
    qty_col = qty_col[qty_col > 0]
    # --------- reindex to a range index --------
    price_col.index = pd.RangeIndex(0, len(price_col), 1)
    qty_col.index = pd.RangeIndex(0, len(qty_col), 1)
    time_col.index = pd.RangeIndex(0, len(time_col), 1)
    ask_col.index = pd.RangeIndex(0, len(qty_col), 1)
    bid_col.index = pd.RangeIndex(0, len(qty_col), 1)
    # -------- find bar features -------
    open_price = price_col.groupby(price_col.index // bar_size).first()
    close_price = price_col.groupby(price_col.index // bar_size).last()
    high_price = price_col.groupby(price_col.index // bar_size).max()
    low_price = price_col.groupby(price_col.index // bar_size).min()
    vol_price_product = price_col.mul(qty_col).groupby(price_col.index // bar_size).sum()
    total_vol = qty_col.groupby(qty_col.index // bar_size).sum()
    vvap_col = vol_price_product.div(total_vol)
    open_ask = ask_col.groupby(price_col.index // bar_size).first()
    close_ask = ask_col.groupby(price_col.index // bar_size).last()
    high_ask = ask_col.groupby(price_col.index // bar_size).max()
    low_ask = ask_col.groupby(price_col.index // bar_size).min()
    average_ask = ask_col.groupby(price_col.index // bar_size).mean()
    open_bid = bid_col.groupby(price_col.index // bar_size).first()
    close_bid = bid_col.groupby(price_col.index // bar_size).last()
    high_bid = bid_col.groupby(price_col.index // bar_size).max()
    low_bid = bid_col.groupby(price_col.index // bar_size).min()
    average_bid = bid_col.groupby(price_col.index // bar_size).mean()
    spread = ask_col - bid_col
    average_bid_ask_spread = spread.groupby(price_col.index // bar_size).mean()
    std_bid_ask_spread = spread.groupby(price_col.index // bar_size).std()
    # ------- process time column -------
    time_col = time_col.groupby(time_col.index // bar_size).first()
    # ------- create and save the new data frame -------
    new_df = pd.DataFrame({
        "open": open_price,
        "close": close_price,
        "high": high_price,
        "low": low_price,
        "VVAP": vvap_col,
        "volume": total_vol,
        "timestamp": time_col,
        "open_bid": open_bid, "high_bid": high_bid, "low_bid": low_bid, "close_bid": close_bid,
        "average_bid": average_bid,
        "open_ask": open_ask, "close_ask": close_ask, "high_ask": high_ask, "low_ask": low_ask,
        "average_ask": average_ask,
        "average_bid_ask_spread": average_bid_ask_spread, "std_bid_ask_spread": std_bid_ask_spread})
    new_df["tick_time"] = [*range(0, len(new_df) * bar_size, bar_size)]
    new_df["timestamp"] = new_df["timestamp"].mask(lambda x: x == 0).interpolate().ffill().bfill()
    if save:
        new_df.to_csv(save, index=False)
    end_time = time.time()
    print("Tick sampling completed : elasped time : " + str(end_time - start_time))
    return new_df
