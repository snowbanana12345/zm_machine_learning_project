class ArrayLengthMisMatchException(Exception):
    def __init__(self, array_1_length : int, array_2_length : int, array_1_description : str, array_2_description : str):
        message = "Array mismatch -- " + array_1_description + " of length : " + str(array_1_length)
        message += " -- " + array_2_description + " of length : " + str(array_2_length)
        super().__init__(message)


class IncorrectDataTypeException(Exception):
    def __init__(self, expected : type, actual : type):
        message = "Expected : " + str(expected) + " : Actual : " + str(actual)
        super().__init__(message)