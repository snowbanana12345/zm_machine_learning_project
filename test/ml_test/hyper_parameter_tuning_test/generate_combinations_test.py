from typing import Dict, List, Any,Iterator
import src.machine_learning_module.utils.hyper_parameter_tuning as hyper_mod

possible_param_values : Dict[str, List[Any]] = {
    "penalty" : ['l1', 'l2', 'elasticnet', 'none'],
    'tol' : [1e-3, 1e-4, 1e-5],
    'C' : [1.5, 1.0, 0.5, 0.25, 0.1],
    'fit_intercept' : [True, False],
    'class_weights' : ['balanced'],
    'solver' : ['saga'],
}

para_combs_it : Iterator[Dict[str, Any]] = hyper_mod.generate_combinations(possible_param_values)
for para in para_combs_it:
    print(para)

