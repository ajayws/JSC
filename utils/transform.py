import numpy as np


def series_to_features_matrix(series, look_back):
    """
    Transform series data into matrix
    :param
        series: array/list to matrix of previous time steps
        look_back: previous time steps used for predict the next period
    :return:
        f_matrix : features matrix
                numpy array (len(series), look_back)
                shape(nb of samples, nb of features)
        y : response of the features
    """
    x, y = [], []
    for i in range(len(series) - look_back - 1):
        a = series[i:(i + look_back)]
        x.append(a)
        y.append(series[i + look_back])
    return np.array(x), np.array(y)


def features_matrix_to_rnn_matrix(fmatrix, use_time_step=True):
    """
    LSTM RNN required an input with shape (nb of samples, nb of look back,
    nb of features). This function is used to transform features
    matrix (nb of samples, nb of features) into RNN matrix.
    If use_time_step equals to True, instead of phrasing the past
    observations as separate input features, we can use them as
    time steps of the ONE input feature.
    ex :
    series = [1,2,3,4,5]
    look back = 2
    fmatrix = [[1,2],
                [2,3],
                [3,4],
                [4,5]]
    RNN Matrix = [[[1,2]],   --> use_time_step = False
                    [[2,3]],
                    [[3,4]],
                    [[4,5]]]
    RNN Matrix = [[[1],[2]],   --> use_time_step = True
                    [[2],[3]],
                    [[3],[4]],
                    [[4],[5]]]
    RNN Matrix
    :param
        f_matrix : features matrix
                numpy array (len(series), look_back)
                shape(nb of samples, nb of features)
        use_time_step : (boolean) True
    :return:
        rnn_matrix : numpy array
                shape(nb of samples, nb of time step, nb of features)
    """
    if use_time_step:
        rnn_matrix = np.reshape(fmatrix, (fmatrix.shape[0], fmatrix.shape[1], 1))
    else:
        rnn_matrix = np.reshape(fmatrix, (fmatrix.shape[0], 1, fmatrix.shape[1]))
    return rnn_matrix
