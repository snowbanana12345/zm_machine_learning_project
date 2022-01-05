from typing import List, Dict
import talib

ATR = "talib_atr"
NATR = "talib_natr"
TRANGE = "talib_trange"

FEATURE_NAME_LIST: List[str] = [ATR, NATR, TRANGE]

# ------ generator functions ------
FEATURE_CREATION_FUNCTION_DICT: Dict[str, callable] = {
    ATR : lambda open, close, high, low, timeperiod : talib.ATR(high, low, close, timeperiod = timeperiod),
    NATR : lambda open, close, high, low, timeperiod : talib.NATR(high, low, close, timeperiod = timeperiod),
    TRANGE : lambda open, high, low, close : talib.TRANGE(high, low, close)
}

# ----- arguments for each feature -------
FEATURE_ARGUMENTS_DICT: Dict[str, List[str]] = {
    ATR : ["timeperiod"],
    NATR : ["timeperiod"],
    TRANGE : []
}

# ------ validation functions --------
ARGUMENT_VALIDATION_FUNC_DICT: Dict[str, callable] = {
    ATR : lambda timeperiod : (timeperiod > 0),
    NATR : lambda timeperiod : (timeperiod > 0),
    TRANGE : lambda : True,
}

# ------ notes on what arguments works -------
FEATURE_ARGUMENT_NOTES: Dict[str, str] = {
    ATR : "[timeperiod > 0]",
    NATR : "[timeperiod > 0]",
    TRANGE : "No Arguments"
}

# ------ window sizes ------
FEATURE_WINDOW_SIZE_FUNC_DICT: Dict[str, callable] = {
    ATR : lambda timeperiod : timeperiod,
    NATR : lambda timeperiod : timeperiod,
    TRANGE : lambda : 1,
}

