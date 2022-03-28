from typing import List, Dict, Tuple
from src.machine_learning_module.utils.random_feature_generators import RandomArgumentGen
import talib

ADX : str = "talib_adx"
ADXR : str = "talib_adxr"
APO : str = "talib_apo"
AROON_UP : str = "talib_aroon_up"
AROON_DOWN : str = "talib_aroon_down"
AROONOSC : str = "talib_aroonosc"
CCI : str = "talib_cci"
CMO : str = "talib_cmo"
MACD : str = "talib_macd"
MACD_SIGNAL : str = "talib_macd_signal"
MACD_HIST : str = "talib_macd_hist"
MOM : str = "talib_mom"
PPO : str = "talib_ppo"
ROC : str = "talib_roc"
ROCP : str = "talib_rocp"
RSI : str = "talib_rsi"
STOCH_SLOWK : str = "talib_stoch_slowk"
STOCH_SLOWD : str = "talib_stoch_slowd"
STOCHF_FASTK : str = "talib_stochf_fastk"
STOCHF_FASTD : str = "talib_stochf_fastd"
TRIX : str = "talib_trix"
ULTOSC : str = "talib_ultosc"
WILLR : str = "talib_willr"

FEATURE_NAME_LIST: List[str] = [ADX, ADXR, APO, AROON_UP, AROON_DOWN, AROONOSC, CCI, CMO, MACD, MACD_SIGNAL,
                                MACD_HIST, MOM, PPO, ROC, ROCP, RSI, STOCH_SLOWK, STOCH_SLOWD, STOCHF_FASTK,
                                STOCHF_FASTD, TRIX, ULTOSC, WILLR]

# ------ generator functions ------
""" takes in pandas series / np.array open close high low close and some other arguments """
FEATURE_CREATION_FUNCTION_DICT: Dict[str, callable] = {
    ADX: lambda open, close, high, low, timeperiod: talib.ADX(high, low, close, timeperiod=timeperiod),
    ADXR: lambda open, close, high, low, timeperiod: talib.ADXR(high, low, close, timeperiod=timeperiod),
    APO: lambda open, close, high, low, fastperiod, slowperiod: talib.APO(close, fastperiod=fastperiod,
                                                                          slowperiod=slowperiod),
    AROON_UP: lambda open, close, high, low, timeperiod: talib.AROON(high, low, timeperiod=timeperiod)[0],
    AROON_DOWN: lambda open, close, high, low, timeperiod: talib.AROON(high, low, timeperiod=timeperiod)[1],
    AROONOSC: lambda open, close, high, low, timeperiod: talib.AROONOSC(high, low, timeperiod=timeperiod),
    CCI: lambda open, close, high, low, timeperiod: talib.CCI(high, low, close, timeperiod=timeperiod),
    CMO: lambda open, close, high, low, timeperiod: talib.CMO(close, timeperiod=timeperiod),
    MACD: lambda open, close, high, low, fastperiod, slowperiod, signalperiod:
    talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[0],
    MACD_SIGNAL: lambda open, close, high, low, fastperiod, slowperiod, signalperiod:
    talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[1],
    MACD_HIST: lambda open, close, high, low, fastperiod, slowperiod, signalperiod:
    talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[2],
    MOM: lambda open, close, high, low, timeperiod: talib.MOM(close, timeperiod=timeperiod),
    PPO: lambda open, close, high, low, fastperiod, slowperiod: talib.PPO(close, fastperiod=fastperiod,
                                                                          slowperiod=slowperiod),
    ROC: lambda open, close, high, low, timeperiod: talib.ROC(close, timeperiod=timeperiod),
    ROCP: lambda open, close, high, low, timeperiod: talib.ROCP(close, timeperiod=timeperiod),
    RSI: lambda open, close, high, low, timeperiod: talib.RSI(close, timeperiod=timeperiod),
    STOCH_SLOWK: lambda open, close, high, low, fastk_period, slowk_period:
    talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period)[0],
    STOCH_SLOWD: lambda open, close, high, low, fastk_period, slowk_period:
    talib.STOCH(high, low, close, fastk_period=fastk_period, slowk_period=slowk_period)[1],
    STOCHF_FASTK: lambda open, close, high, low, fastk_period, fastd_period:
    talib.STOCHF(high, low, close, fastk_period=fastk_period, fastd_period=fastd_period)[0],
    STOCHF_FASTD: lambda open, close, high, low, fastk_period, fastd_period:
    talib.STOCHF(high, low, close, fastk_period=fastk_period, fastd_period=fastd_period)[1],
    TRIX: lambda open, close, high, low, timeperiod: talib.TRIX(close, timeperiod=timeperiod),
    ULTOSC: lambda open, close, high, low, timeperiod1, timeperiod2, timeperiod3: talib.ULTOSC(high, low, close,
                                                                                               timeperiod1=timeperiod1,
                                                                                               timeperiod2=timeperiod2,
                                                                                               timeperiod3=timeperiod3),
    WILLR: lambda open, close, high, low, timeperiod: talib.WILLR(high, low, close, timeperiod=timeperiod)
}

# ----- arguments for each feature -------
FEATURE_ARGUMENTS_DICT: Dict[str, List[str]] = {
    ADX: ['timeperiod'],
    ADXR: ['timeperiod'],
    APO: ['fastperiod', 'slowperiod'],
    AROON_UP: ['timeperiod'],
    AROON_DOWN: ['timeperiod'],
    AROONOSC: ['timeperiod'],
    CCI: ['timeperiod'],
    CMO: ['timeperiod'],
    MACD: ['fastperiod', 'slowperiod', 'signalperiod'],
    MACD_SIGNAL: ['fastperiod', 'slowperiod', 'signalperiod'],
    MACD_HIST: ['fastperiod', 'slowperiod', 'signalperiod'],
    MOM: ['timeperiod'],
    PPO: ['fastperiod', 'slowperiod'],
    RSI: ['timeperiod'],
    ROC: ['timeperiod'],
    ROCP: ['timeperiod'],
    STOCH_SLOWK: ['fastk_period', 'slowk_period'],
    STOCH_SLOWD: ['fastk_period', "slowk_period"],
    STOCHF_FASTK: ['fastk_period', 'fastd_period'],
    STOCHF_FASTD: ['fastk_period', 'fastd_period'],
    TRIX: ['timeperiod'],
    ULTOSC: ['timeperiod1', 'timeperiod2', 'timeperiod3'],
    WILLR: ['timeperiod']
}

# ------ validation functions --------
ARGUMENT_VALIDATION_FUNC_DICT: Dict[str, callable] = {
    ADX: lambda timeperiod: (timeperiod > 0),
    ADXR: lambda timeperiod: (timeperiod > 0),
    APO: lambda fastperiod, slowperiod: (fastperiod > 0) and (slowperiod > 0),
    AROON_UP: lambda timeperiod: (timeperiod > 0),
    AROON_DOWN: lambda timeperiod: (timeperiod > 0),
    AROONOSC: lambda timeperiod: (timeperiod > 0),
    CCI: lambda timeperiod: (timeperiod > 0),
    CMO: lambda timeperiod: (timeperiod > 0),
    MACD: lambda fastperiod, slowperiod, signalperiod: (fastperiod > 0) and (slowperiod > 0) and (signalperiod > 0),
    MACD_SIGNAL: lambda fastperiod, slowperiod, signalperiod: (fastperiod > 0) and (slowperiod > 0) and (
                signalperiod > 0),
    MACD_HIST: lambda fastperiod, slowperiod, signalperiod: (fastperiod > 0) and (slowperiod > 0) and (
                signalperiod > 0),
    MOM: lambda timeperiod: (timeperiod > 0),
    PPO: lambda fastperiod, slowperiod: (fastperiod > 0) and (slowperiod > 0),
    RSI: lambda timeperiod: (timeperiod > 0),
    ROC: lambda timeperiod: (timeperiod > 0),
    ROCP: lambda timeperiod: (timeperiod > 0),
    STOCH_SLOWK: lambda fastk_period, slowk_period: (fastk_period > 0) and (slowk_period > 0),
    STOCH_SLOWD: lambda fastk_period, slowk_period: (fastk_period > 0) and (slowk_period > 0),
    STOCHF_FASTK: lambda fastk_period, fastd_period: (fastk_period > 0) and (fastd_period > 0),
    STOCHF_FASTD: lambda fastk_period, fastd_period: (fastk_period > 0) and (fastd_period > 0),
    TRIX: lambda timeperiod: (timeperiod > 0),
    ULTOSC: lambda timeperiod1, timeperiod2, timeperiod3: (timeperiod1 > 0) and (timeperiod2 > 0) and (timeperiod3 > 0),
    WILLR: lambda timeperiod: (timeperiod > 1)
}

# ------ notes on what arguments works -------
FEATURE_ARGUMENT_NOTES: Dict[str, str] = {
    ADX: "[timeperiod > 0]",
    ADXR: "[timeperiod > 0]",
    APO: "[fastperiod > 0] and [slowperiod > 0]",
    AROON_UP: "[timeperiod > 0]",
    AROON_DOWN: "[timeperiod > 0]",
    AROONOSC: "[timeperiod > 0]",
    CCI: "[timeperiod > 0]",
    CMO: "[timeperiod > 0]",
    MACD: "[fastperiod > 0] and [slowperiod > 0] and [signalperiod > 0]",
    MACD_SIGNAL: "[fastperiod > 0] and [slowperiod > 0] and [signalperiod > 0]",
    MACD_HIST: "[fastperiod > 0] and [slowperiod > 0] and [signalperiod > 0]",
    MOM: "[timeperiod > 0]",
    PPO: "[timeperiod > 0]",
    ROC: "[timeperiod > 0]",
    RSI : "[timeperiod > 0]",
    ROCP: "[timeperiod > 0]",
    STOCH_SLOWK: "[fastk_period > 0] and [slowk_period > 0]",
    STOCH_SLOWD: "[fastk_period > 0] and [slowk_period > 0]",
    STOCHF_FASTK: "[fastk_period > 0] and [fastd_period > 0]",
    STOCHF_FASTD: "[fastk_period > 0] and [fastd_period > 0]",
    TRIX: "[timeperiod > 0]",
    ULTOSC: "[timeperiod1 > 0] and [timeperiod2 > 0] and [timeperiod3 > 0]",
    WILLR: "[timeperiod > 1]",
}

# ------ window sizes ------
FEATURE_WINDOW_SIZE_FUNC_DICT: Dict[str, callable] = {
    ADX: lambda timeperiod: timeperiod,
    ADXR: lambda timeperiod: timeperiod,
    APO: lambda fastperiod, slowperiod: max(fastperiod, slowperiod),
    AROON_UP: lambda timeperiod: timeperiod,
    AROON_DOWN: lambda timeperiod: timeperiod,
    AROONOSC: lambda timeperiod: timeperiod,
    CCI: lambda timeperiod: timeperiod,
    CMO: lambda timeperiod: timeperiod,
    MACD: lambda fastperiod, slowperiod, signalperiod: max(fastperiod, slowperiod, signalperiod),
    MACD_SIGNAL: lambda fastperiod, slowperiod, signalperiod: max(fastperiod, slowperiod, signalperiod),
    MACD_HIST: lambda fastperiod, slowperiod, signalperiod: max(fastperiod, slowperiod, signalperiod),
    MOM: lambda timeperiod: timeperiod,
    PPO: lambda fastperiod, slowperiod: max(fastperiod, slowperiod),
    RSI: lambda timeperiod: timeperiod,
    ROC: lambda timeperiod: timeperiod,
    ROCP: lambda timeperiod: timeperiod,
    STOCH_SLOWK: lambda fastk_period, slowk_period: max(fastk_period, slowk_period),
    STOCH_SLOWD: lambda fastk_period, slowk_period: max(fastk_period, slowk_period),
    STOCHF_FASTK: lambda fastk_period, fastd_period: max(fastk_period, fastd_period),
    STOCHF_FASTD: lambda fastk_period, fastd_period: max(fastk_period, fastd_period),
    TRIX: lambda timeperiod: timeperiod,
    ULTOSC: lambda timeperiod1, timeperiod2, timeperiod3: max(timeperiod1, timeperiod2, timeperiod3),
    WILLR: lambda timeperiod: timeperiod,
}

# ------ Random Argument generator generators -------
import random

class IntRandomGenUniform(RandomArgumentGen):
    def __init__(self, range_dict : Dict[str, Tuple[int, int]]):
        """
        :param range_dict: Dictionary of the format : Dict[feature name, Tuple[min, max]]
        """
        super().__init__()
        self.range_dict = range_dict

    def generate_arg_dict(self) -> Dict[str, int]:
        new_arg_dict = {}
        for feat_name, (arg_min, arg_max) in self.range_dict.items():
            new_arg_dict[feat_name] = random.randint(arg_min, arg_max)
        return new_arg_dict

UNIFORM_ARG_DICT_GEN_GEN_FUNC : Dict[str, callable] = {
    ADX : lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)}),
    ADXR:  lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)}),
    APO : lambda fast_min, fast_max, slow_min, slow_max : IntRandomGenUniform({'fastperiod' : (fast_min, fast_max), 'slowperiod' : (slow_min, slow_max)}),
    AROON_UP: lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)}),
    AROON_DOWN: lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)}),
    AROONOSC: lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)}),
    CCI: lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)}),
    CMO: lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)}),
    MACD: lambda fast_min, fast_max, slow_min, slow_max, signal_min, signal_max: IntRandomGenUniform({'fastperiod' : (fast_min, fast_max),
                                                            'slowperiod' : (slow_min, slow_max), 'signalperiod' : (signal_min, signal_max)}),
    MACD_SIGNAL:  lambda fast_min, fast_max, slow_min, slow_max, signal_min, signal_max: IntRandomGenUniform({'fastperiod' : (fast_min, fast_max),
                                                            'slowperiod' : (slow_min, slow_max), 'signalperiod' : (signal_min, signal_max)}),
    MACD_HIST: lambda fast_min, fast_max, slow_min, slow_max, signal_min, signal_max: IntRandomGenUniform({'fastperiod' : (fast_min, fast_max),
                                                            'slowperiod' : (slow_min, slow_max), 'signalperiod' : (signal_min, signal_max)}),
    MOM: lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)}),
    PPO: lambda fast_min, fast_max, slow_min, slow_max : IntRandomGenUniform({'fastperiod' : (fast_min, fast_max), 'slowperiod' : (slow_min, slow_max)}),
    RSI: lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)}),
    ROC: lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)}),
    ROCP: lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)}),
    STOCH_SLOWK: lambda fastk_min, fastk_max, slowk_min, slowk_max: IntRandomGenUniform({"fastk_period" : (fastk_min, fastk_max), "slowk_period" : (slowk_min, slowk_max)}),
    STOCH_SLOWD: lambda fastk_min, fastk_max, slowk_min, slowk_max: IntRandomGenUniform({"fastk_period" : (fastk_min, fastk_max), "slowk_period" : (slowk_min, slowk_max)}),
    STOCHF_FASTK: lambda fastk_min, fastk_max, fastd_min, fastd_max: IntRandomGenUniform({"fastk_period" : (fastk_min, fastk_max), "fastd_period" : (fastd_min, fastd_max)}),
    STOCHF_FASTD: lambda fastk_min, fastk_max, fastd_min, fastd_max: IntRandomGenUniform({"fastk_period" : (fastk_min, fastk_max), "fastd_period" : (fastd_min, fastd_max)}),
    TRIX: lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)}),
    ULTOSC: lambda tp1_min, tp1_max, tp2_min, tp2_max, tp3_min, tp3_max : IntRandomGenUniform({'timeperiod1' : (tp1_min, tp1_max),
                                                                        'timeperiod2' : (tp2_min, tp2_max), 'timeperiod3' : (tp3_min, tp3_max)}),
    WILLR: lambda timeperiod_min, timeperiod_max : IntRandomGenUniform({'timeperiod' : (timeperiod_min, timeperiod_max)})
}



