from typing import Dict, List, Any, Iterator
import itertools

def generate_combinations(param_values_dict : Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    names : List[str] = sorted(param_values_dict)
    combinations = itertools.product(*[param_values_dict[name] for name in names])
    return map(lambda comb : {key:val for key,val in zip(names,comb)}, combinations)


class ParamDictValidator:
    def __init__(self):
        pass

    def validate_values(self) -> bool:
        return True

