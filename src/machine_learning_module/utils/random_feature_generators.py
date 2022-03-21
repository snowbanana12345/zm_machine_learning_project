from typing import Dict, Any, List

class RandomArgumentGen:
    """
    Abstract class, implement this for each feature
    """
    def __init__(self):
        pass

    def generate_arg_dict(self) -> Dict[str, Any]:
        return {}


class RandomParaDictGenerator:
    def __init__(self, generator_dict : Dict[str, RandomArgumentGen], repeats_dict : Dict[str, int]):
        """
        :param generator_dict: Dictionary of the form Dict[feature name, arguments dict generator]
        :param repeats_dict: Dictionary of the form Dict[feature name, number of features to generate]
        NOTE : arguments dict generators are assumed to be infinite, will cause crashes iterator is used too many times
        NOTE : this implementation does not guarantee uniqueness of the feature arguments generated
        NOTE : keys of generator_dict and repeats_dict has to be the same
        """
        # ---- validate inputs -----
        if not generator_dict.keys() == repeats_dict.keys():
            raise ValueError("Feature names of generator_dict and repeat_dict are not equal")
        for feat_name, repeats in repeats_dict.items():
            if repeats <= 0:
                raise ValueError("Number of repeats must be positive!")
        # ---- store -----
        self.generator_dict : Dict[str, RandomArgumentGen] = generator_dict
        self.repeats_dict : Dict[str, int] = repeats_dict

    def generate_param_dict(self) -> Dict[str, List[Dict[str, Any]]] :
        new_param_dict : Dict[str, List[Dict[str, Any]]] = {feat_name : [] for feat_name in self.generator_dict}
        for feat_name in self.generator_dict.keys():
            num_repeats : int = self.repeats_dict[feat_name]
            arg_dict_gen : RandomArgumentGen = self.generator_dict[feat_name]
            for _ in range(num_repeats):
                new_param_dict[feat_name].append(arg_dict_gen.generate_arg_dict())
        return new_param_dict


def create_random_param_gen(arg_dict_gen_param_dict : Dict[str, Dict[str, Any]],
                            feature_name_list : List[str],
                            arg_dict_gen_func : Dict[str, callable],
                            repeats_dict : Dict[str, int]) -> RandomParaDictGenerator:
    arg_dict_gen_dict: Dict[str, RandomArgumentGen] = {}
    for feat_name in feature_name_list:
        arg_dict_gen_dict[feat_name] = arg_dict_gen_func[feat_name](**arg_dict_gen_param_dict[feat_name])
    return RandomParaDictGenerator(generator_dict=arg_dict_gen_dict, repeats_dict=repeats_dict)


