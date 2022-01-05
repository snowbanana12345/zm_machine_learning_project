from src.machine_learning_module.utils.random_feature_generators import RandomArgumentGen, RandomParaDictGenerator
import src.machine_learning_module.feature_generator_data.talib_momentum as talib_mom_data
from typing import Dict, List, Any

# ---- random gen feature parameters -----
arg_dict_gen_param_dict : Dict[str, Dict[str, int]] = {
    talib_mom_data.ADX : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.ADXR : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.APO : {"fast_min" : 10, "fast_max" : 30, "slow_min" : 10, "slow_max" : 30},
    talib_mom_data.AROON_UP : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.AROON_DOWN :  {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.AROONOSC : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.CCI : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.CMO : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.MACD: {"fast_min" : 10, "fast_max" : 30, "slow_min" : 10, "slow_max" : 30, "signal_min" : 10, "signal_max" : 30},
    talib_mom_data.MACD_SIGNAL : {"fast_min" : 10, "fast_max" : 30, "slow_min" : 10, "slow_max" : 30, "signal_min" : 10, "signal_max" : 30},
    talib_mom_data.MACD_HIST: {"fast_min" : 10, "fast_max" : 30, "slow_min" : 10, "slow_max" : 30, "signal_min" : 10, "signal_max" : 30},
    talib_mom_data.MOM : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.PPO : {"fast_min" : 10, "fast_max" : 30, "slow_min" : 10, "slow_max" : 30},
    talib_mom_data.ROC : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.ROCP: {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.RSI : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.STOCH_SLOWK : {'fastk_min' : 10, 'fastk_max' : 30, 'slowk_min' : 10, "slowk_max" : 30},
    talib_mom_data.STOCH_SLOWD : {'fastk_min' : 10, 'fastk_max' : 30, 'slowk_min' : 10, "slowk_max" : 30},
    talib_mom_data.STOCHF_FASTK : {'fastk_min' : 10, 'fastk_max' : 30, 'fastd_min' : 10, "fastd_max" : 30},
    talib_mom_data.STOCHF_FASTD : {'fastk_min' : 10, 'fastk_max' : 30, 'fastd_min' : 10, "fastd_max" : 30},
    talib_mom_data.TRIX : {'timeperiod_min' : 10, 'timeperiod_max' : 30},
    talib_mom_data.ULTOSC : {'tp1_min' : 10, 'tp1_max' : 30, 'tp2_min' : 10, 'tp2_max' : 30, 'tp3_min' : 10, "tp3_max" : 30},
    talib_mom_data.WILLR : {'timeperiod_min' : 10, 'timeperiod_max' : 30}
}

repeats_dict = {
    talib_mom_data.ADX: 2,
    talib_mom_data.ADXR: 2,
    talib_mom_data.APO: 2,
    talib_mom_data.AROON_UP:  2,
    talib_mom_data.AROON_DOWN: 2,
    talib_mom_data.AROONOSC: 2,
    talib_mom_data.CCI: 2,
    talib_mom_data.CMO: 2,
    talib_mom_data.MACD: 2,
    talib_mom_data.MACD_SIGNAL: 2,
    talib_mom_data.MACD_HIST: 2,
    talib_mom_data.MOM: 2,
    talib_mom_data.PPO: 2,
    talib_mom_data.ROC: 2,
    talib_mom_data.ROCP: 2,
    talib_mom_data.RSI: 2,
    talib_mom_data.STOCH_SLOWK: 2,
    talib_mom_data.STOCH_SLOWD: 2,
    talib_mom_data.STOCHF_FASTK: 2 ,
    talib_mom_data.STOCHF_FASTD: 2,
    talib_mom_data.TRIX: 2,
    talib_mom_data.ULTOSC: 2,
    talib_mom_data.WILLR: 2
}

arg_dict_gen_dict  : Dict[str, RandomArgumentGen] = {}
for feat_name in talib_mom_data.FEATURE_NAME_LIST:
    arg_dict_gen : RandomArgumentGen = talib_mom_data.UNIFORM_ARG_DICT_GEN_GEN_FUNC[feat_name](**arg_dict_gen_param_dict[feat_name])
    arg_dict_gen_dict[feat_name] = arg_dict_gen

param_dict_gen : RandomParaDictGenerator = RandomParaDictGenerator(generator_dict = arg_dict_gen_dict, repeats_dict = repeats_dict)
para_dict : Dict[str, List[Dict[str, Any]]] = param_dict_gen.generate_param_dict()

for feat_name, arg_dict_lst in para_dict.items():
    print(f"Feature : {feat_name}")
    for arg_dict in arg_dict_lst:
        print(arg_dict)


