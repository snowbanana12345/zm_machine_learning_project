from typing import List, Dict
import talib

# ------ feature names -------
CDL2CROWS = "talib_cdl_2_crows"
CDL3BLACKCROWS = "talib_cdl_3_blackcrows"
CDL3INSIDE = "talib_cdl_3_inside"
CDL3LINESTRIKE = "talib_cdl_3_linestrike"
CDL3OUTSIDE = "talib_cdl_3_outside"
CDL3STARSINSOUTH = "talib_cdl_3_starsinsouth"
CDL3WHITESOLDIERS = "talib_cdl_3_whitesoldiers"
CDLABANDONEDBABY = "talib_cdl_abandoned_baby"
CDLADVANCEBLOCK = "talib_cdl_advance_block"
CDLBELTHOLD = "talib_cdl_belthold"
CDLBREAKAWAY = "talib_cdl_breakaway"
CDLCLOSINGMARUBOZU = "talib_cdl_closing_marubozu"
CDLCONCEALBABYSWALL = "talib_cdl_conceal_baby_swall"
CDLCOUNTERATTACK = "talib_cdl_counterattack"
CDLDARKCLOUDCOVER = "talib_cdl_dark_cloud_cover"
CDLDOJI = "talib_cdl_doji"
CDLDOJISTAR = "talib_cdl_doji_star"
CDLDRAGONFLYDOJI = "talib_cdl_dragonfly_doji"
CDLENGULFING = "talib_cdl_engulfing"
CDLEVENINGDOJISTAR = "talib_cdl_evening_dojistar"
CDLGAPSIDESIDEWHITE = "talib_cdl_gap_side_side_white"
CDLGRAVESTONEDOJI = "talib_cdl_gravestone_doji"
CDLHAMMER = "talib_cdl_hammer"
CDLHANGINGMAN = "talib_cdl_hanging_man"
CDLHARAMI = "talib_cdl_harami"
CDLHARAMICROSS = "talib_cdl_haramicross"
CDLHIGHWAVE = "talib_cdl_high_wave"
CDLHIKKAKE = "talib_cdl_hikkake"
CDLHIKKAKEMOD = "talib_cdl_hikkakemod"
CDLHOMINGPIGEON = "talib_cdl_homing_pigeon"
CDLIDENTICAL3CROWS = "talib_cdl_identical_3_crows"
CDLINNECK = "talib_cdl_in_neck"
CDLINVERTEDHAMMER = "talib_cdl_inverted_hammer"
CDLKICKING = "talib_cdl_kicking"
CDLKICKINGBYLENGTH = "talib_cdl_kicking_by_length"
CDLLADDERBOTTOM = "talib_cdl_labber_bottom"
CDLLONGLEGGEDDOJI = "talib_cdl_long_legged_doji"
CDLLONGLINE = "talib_cdl_long_line"
CDLMARUBOZU = "talib_cdl_marubozu"
CDLMATCHINGLOW = "talib_cdl_matching_low"
CDLMATHOLD = "talib_cdl_mat_hold"
CDLMORNINGDOJISTAR = "talib_cdl_morning_doji_star"
CDLMORNINGSTAR = "talib_cdl_morning_star"
CDLONNECK = "talib_cdl_on_neck"
CDLPIERCING = "talib_cdl_piercing"
CDLRICKSHAWMAN = "talib_cdl_rick_shawman"
CDLRISEFALL3METHODS = "talib_cdl_rise_fall_3_methods"
CDLSEPARATINGLINES = "talib_cdl_separating_lines"
CDLSHOOTINGSTAR = "talib_cdl_shooting_star"
CDLSHORTLINE = "talib_cdl_shortline"
CDLSPINNINGTOP = "talib_cdl_spinning_top"
CDLSTALLEDPATTERN = "talib_cdl_stalled_pattern"
CDLSTICKSANDWICH = "talib_cdl_stick_sandwich"
CDLTAKURI = "talib_cdl_takuri"
CDLTASUKIGAP = "talib_cdl_tasukigap"
CDLTHRUSTING = "talib_cdl_thrusting"
CDLTRISTAR = "talib_cdl_tristar"
CDLUNIQUE3RIVER = "talib_cdl_unique_3_river"
CDLUPSIDEGAP2CROWS = "talib_cdl_upside_gap_2_crows"
CDLXSIDEGAP3METHODS = "talib_cdl_xside_gap_3_methods"

# ----- feature name list ------
FEATURE_NAME_LIST : List[str] = [
    CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE ,CDL3LINESTRIKE,CDL3OUTSIDE,CDL3STARSINSOUTH,CDL3WHITESOLDIERS,CDLABANDONEDBABY,
    CDLADVANCEBLOCK ,CDLBELTHOLD,CDLBREAKAWAY,CDLCLOSINGMARUBOZU,CDLCONCEALBABYSWALL,CDLCOUNTERATTACK,CDLDARKCLOUDCOVER,CDLDOJI,
    CDLDOJISTAR,CDLDRAGONFLYDOJI,CDLENGULFING,CDLEVENINGDOJISTAR,CDLGAPSIDESIDEWHITE,CDLGRAVESTONEDOJI,CDLHAMMER,CDLHANGINGMAN ,CDLHARAMI,CDLHARAMICROSS,
    CDLHIGHWAVE,CDLHIKKAKE,CDLHIKKAKEMOD,CDLHOMINGPIGEON ,CDLIDENTICAL3CROWS ,CDLINNECK,CDLINVERTEDHAMMER,CDLKICKING ,CDLKICKINGBYLENGTH ,CDLLADDERBOTTOM,
    CDLLONGLEGGEDDOJI ,CDLLONGLINE,CDLMARUBOZU,CDLMATCHINGLOW ,CDLMATHOLD,CDLMORNINGDOJISTAR,CDLMORNINGSTAR,CDLONNECK,CDLPIERCING,
    CDLRICKSHAWMAN ,CDLRISEFALL3METHODS ,CDLSEPARATINGLINES ,CDLSHOOTINGSTAR ,CDLSHORTLINE ,CDLSPINNINGTOP,CDLSTALLEDPATTERN,
    CDLSTICKSANDWICH,CDLTAKURI,CDLTASUKIGAP,CDLTHRUSTING ,CDLTRISTAR,CDLUNIQUE3RIVER ,CDLUPSIDEGAP2CROWS ,CDLXSIDEGAP3METHODS]

# ------ generator functions ------
FEATURE_CREATION_FUNCTION_DICT: Dict[str, callable] = {
    CDL2CROWS: lambda open, close, high, low: talib.CDL2CROWS(open, high, low, close),
    CDL3BLACKCROWS: lambda open, close, high, low: talib.CDL3BLACKCROWS(open, high, low, close),
    CDL3INSIDE: lambda open, close, high, low: talib.CDL3INSIDE(open, high, low, close),
    CDL3LINESTRIKE: lambda open, close, high, low: talib.CDL3LINESTRIKE(open, high, low, close),
    CDL3OUTSIDE: lambda open, close, high, low: talib.CDL3OUTSIDE(open, high, low, close),
    CDL3STARSINSOUTH: lambda open, close, high, low: talib.CDL3STARSINSOUTH(open, high, low, close),
    CDL3WHITESOLDIERS: lambda open, close, high, low: talib.CDL3WHITESOLDIERS(open, high, low, close),
    CDLABANDONEDBABY: lambda open, close, high, low: talib.CDLABANDONEDBABY(open, high, low, close),
    CDLADVANCEBLOCK: lambda open, close, high, low: talib.CDLADVANCEBLOCK(open, high, low, close),
    CDLBELTHOLD: lambda open, close, high, low: talib.CDLBELTHOLD(open, high, low, close),
    CDLBREAKAWAY: lambda open, close, high, low: talib.CDLBREAKAWAY(open, high, low, close),
    CDLCLOSINGMARUBOZU: lambda open, close, high, low: talib.CDLCLOSINGMARUBOZU(open, high, low, close),
    CDLCONCEALBABYSWALL: lambda open, close, high, low: talib.CDLCONCEALBABYSWALL(open, high, low, close),
    CDLCOUNTERATTACK: lambda open, close, high, low: talib.CDLCOUNTERATTACK(open, high, low, close),
    CDLDARKCLOUDCOVER: lambda open, close, high, low: talib.CDLDARKCLOUDCOVER(open, high, low, close),
    CDLDOJI: lambda open, close, high, low: talib.CDLDOJI(open, high, low, close),
    CDLDOJISTAR: lambda open, close, high, low: talib.CDLDOJISTAR(open, high, low, close),
    CDLDRAGONFLYDOJI: lambda open, close, high, low: talib.CDLDRAGONFLYDOJI(open, high, low, close),
    CDLENGULFING: lambda open, close, high, low: talib.CDLENGULFING(open, high, low, close),
    CDLEVENINGDOJISTAR: lambda open, close, high, low: talib.CDLEVENINGDOJISTAR(open, high, low, close),
    CDLGAPSIDESIDEWHITE: lambda open, close, high, low: talib.CDLGAPSIDESIDEWHITE(open, high, low, close),
    CDLGRAVESTONEDOJI: lambda open, close, high, low: talib.CDLGRAVESTONEDOJI(open, high, low, close),
    CDLHAMMER: lambda open, close, high, low: talib.CDLHAMMER(open, high, low, close),
    CDLHANGINGMAN: lambda open, close, high, low: talib.CDLHANGINGMAN(open, high, low, close),
    CDLHARAMI: lambda open, close, high, low: talib.CDLHARAMI(open, high, low, close),
    CDLHARAMICROSS: lambda open, close, high, low: talib.CDLHARAMICROSS(open, high, low, close),
    CDLHIGHWAVE: lambda open, close, high, low: talib.CDLHIGHWAVE(open, high, low, close),
    CDLHIKKAKE: lambda open, close, high, low: talib.CDLHIKKAKE(open, high, low, close),
    CDLHIKKAKEMOD: lambda open, close, high, low: talib.CDLHIKKAKEMOD(open, high, low, close),
    CDLHOMINGPIGEON: lambda open, close, high, low: talib.CDLHOMINGPIGEON(open, high, low, close),
    CDLIDENTICAL3CROWS: lambda open, close, high, low: talib.CDLIDENTICAL3CROWS(open, high, low, close),
    CDLINNECK: lambda open, close, high, low: talib.CDLINNECK(open, high, low, close),
    CDLINVERTEDHAMMER: lambda open, close, high, low: talib.CDLINVERTEDHAMMER(open, high, low, close),
    CDLKICKING: lambda open, close, high, low: talib.CDLKICKING(open, high, low, close),
    CDLKICKINGBYLENGTH: lambda open, close, high, low: talib.CDLKICKINGBYLENGTH(open, high, low, close),
    CDLLADDERBOTTOM: lambda open, close, high, low: talib.CDLLADDERBOTTOM(open, high, low, close),
    CDLLONGLEGGEDDOJI: lambda open, close, high, low: talib.CDLLONGLEGGEDDOJI(open, high, low, close),
    CDLLONGLINE: lambda open, close, high, low: talib.CDLLONGLINE(open, high, low, close),
    CDLMARUBOZU: lambda open, close, high, low: talib.CDLMARUBOZU(open, high, low, close),
    CDLMATCHINGLOW: lambda open, close, high, low: talib.CDLMATCHINGLOW(open, high, low, close),
    CDLMATHOLD: lambda open, close, high, low: talib.CDLMATHOLD(open, high, low, close),
    CDLMORNINGDOJISTAR: lambda open, close, high, low: talib.CDLMORNINGDOJISTAR(open, high, low, close),
    CDLMORNINGSTAR: lambda open, close, high, low: talib.CDLMORNINGSTAR(open, high, low, close),
    CDLONNECK: lambda open, close, high, low: talib.CDLONNECK(open, high, low, close),
    CDLPIERCING: lambda open, close, high, low: talib.CDLPIERCING(open, high, low, close),
    CDLRICKSHAWMAN: lambda open, close, high, low: talib.CDLRICKSHAWMAN(open, high, low, close),
    CDLRISEFALL3METHODS: lambda open, close, high, low: talib.CDLRISEFALL3METHODS(open, high, low, close),
    CDLSEPARATINGLINES: lambda open, close, high, low: talib.CDLSEPARATINGLINES(open, high, low, close),
    CDLSHOOTINGSTAR: lambda open, close, high, low: talib.CDLSHOOTINGSTAR(open, high, low, close),
    CDLSHORTLINE: lambda open, close, high, low: talib.CDLSHORTLINE(open, high, low, close),
    CDLSPINNINGTOP: lambda open, close, high, low: talib.CDLSPINNINGTOP(open, high, low, close),
    CDLSTALLEDPATTERN: lambda open, close, high, low: talib.CDLSTALLEDPATTERN(open, high, low, close),
    CDLSTICKSANDWICH: lambda open, close, high, low: talib.CDLSTICKSANDWICH(open, high, low, close),
    CDLTAKURI: lambda open, close, high, low: talib.CDLTAKURI(open, high, low, close),
    CDLTASUKIGAP: lambda open, close, high, low: talib.CDLTASUKIGAP(open, high, low, close),
    CDLTHRUSTING: lambda open, close, high, low: talib.CDLTHRUSTING(open, high, low, close),
    CDLTRISTAR: lambda open, close, high, low: talib.CDLTRISTAR(open, high, low, close),
    CDLUNIQUE3RIVER: lambda open, close, high, low: talib.CDLUNIQUE3RIVER(open, high, low, close),
    CDLUPSIDEGAP2CROWS: lambda open, close, high, low: talib.CDLUPSIDEGAP2CROWS(open, high, low, close),
    CDLXSIDEGAP3METHODS: lambda open, close, high, low: talib.CDLXSIDEGAP3METHODS(open, high, low, close),
}

# ----- arguments for each feature -------
FEATURE_ARGUMENTS_DICT: Dict[str, List[str]] = {feature_name : [] for feature_name in FEATURE_NAME_LIST}

# ------ validation functions --------
ARGUMENT_VALIDATION_FUNC_DICT: Dict[str, callable] = {feature_name : (lambda : True) for feature_name in FEATURE_NAME_LIST}

# ------ notes on what arguments works -------
FEATURE_ARGUMENT_NOTES: Dict[str, str] = {feature_name : "No Arguments" for feature_name in FEATURE_NAME_LIST}

# ------ window sizes ------
FEATURE_WINDOW_SIZE_FUNC_DICT: Dict[str, callable] = {feature_name : (lambda : 1) for feature_name in FEATURE_NAME_LIST}

