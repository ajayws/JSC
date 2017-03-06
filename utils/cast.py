import numpy as np


def float_or_nan(value):
    """
    string to float
    :param
        value: (str) string of number or ''
    :return:
        (float) or numpy.nan
    """
    return float(value if value != '' else np.nan)


if __name__ == '__main__':
    print(float_or_nan(''))
    print(float_or_nan('100'))
