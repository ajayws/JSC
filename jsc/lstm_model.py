import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from utils import transform


class LstmModel(object):
    def __init__(self, series):
        # normalize the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.series = self.scaler.fit_transform(series.reshape(-1, 1))

    def transform_split(self, look_back=5, train_composition=0.7, use_time_step=True):
        """

        :param:
            look_back: (int) how many time steps required to predict
                            future value
            train_composition: (float > 0) percentage or index
            use_time_step : (boolean) True
        :return:
            train_x : rnn_matrix(see transform.py doc) training data
            train_y : rnn_matrix(see transform.py doc) label data
            test_x : 1D array test data
            test_y : 1D array label test data
            train_size : training size
        """
        if train_composition > 1 or train_composition < 0:
            raise Exception('Train composition must be between 0-1')
        train_size = int(len(self.series) * train_composition)
        train = self.series[0:train_size]
        test = self.series[train_size:]

        # split the data test and train
        train_x, train_y = transform.series_to_features_matrix(train, look_back)
        test_x, test_y = transform.series_to_features_matrix(test, look_back)

        # transform into rnn matrix format
        train_x = transform.features_matrix_to_rnn_matrix(train_x, use_time_step)
        test_x = transform.features_matrix_to_rnn_matrix(test_x, use_time_step)
        return train_x, train_y, test_x, test_y, train_size

    @staticmethod
    def build_model(layers):
        """
        Build LSTM Model consisting 4 layers
        :param
            layers : array of network structure
                    [input dim, LSTM 1st layer,
                    LSTM second layer, normal layer(final output) to give
                    the prediction]
        :return:
            model : LSTM keras model
        """
        model = Sequential()

        model.add(LSTM(
            input_dim=layers[0],
            output_dim=layers[1],
            return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(
            layers[2],
            return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(
            output_dim=layers[3]))

        model.add(Activation('tanh'))

        model.compile(loss='mae', optimizer='adadelta')
        return model

    @staticmethod
    def build_model2():
        model = Sequential()
        model.add(LSTM(4, input_dim=1))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    @staticmethod
    def build_model3(layers):
        """
        Build LSTM Model consisting 5 layers
        :param
            layers : array of network structure
                    [input dim, LSTM 1st layer,
                    LSTM second layer, normal layer(final output) to give
                    the prediction]
        :return:
            model : LSTM keras model
        """
        model = Sequential()

        model.add(LSTM(
            input_dim=layers[0],
            output_dim=layers[1],
            return_sequences=True))

        model.add(Dropout(0.2))

        model.add(LSTM(
            layers[2],
            return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(
            layers[3],
            return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(
            output_dim=layers[4]))

        model.add(Activation('tanh'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    @staticmethod
    def fit(model, train_x, train_y, nb_epoch=100, batch_size=1, verbose=2):
        """
        Fit function to train our model
        :param
            model : LSTM keras model
            train_x : rnn_matrix (nb of samples, nb of time step,
                        nb of features) - training data
            train_y : 1D array label data
            nb_epoch : number of epoch (keras parameter)
            batch_size : batch size (keras parameter)
            verbose : (keras parameter)
        :return:
            model : LSTM keras model
        """
        return model.fit(train_x, train_y, nb_epoch=nb_epoch,
                         batch_size=batch_size, verbose=verbose)

    def predict(self, model, test_x):
        """
        Use this function to predict only one single point(value at t+1)
        based on prior true history
        :param
            model : LSTM keras model
            test_x : rnn_matrix (testing data)
        :return:
            test_y = 1D array label data
        """
        test_predict = model.predict(test_x)
        return self.scaler.inverse_transform(test_predict.reshape(-1, 1))

    def predict_future(self, model, test_x):
        """
        Use this function to predict multiple value in the future
        at time t+1, t+2, t+3, etc
        :param
            model : LSTM keras model
            test_x : rnn_matrix (testing data)
        :return:
            test_y = 1D array label data

        """
        test_predict = []
        data = np.array([test_x[0]])
        nb_samples, row, col = test_x.shape
        for i in range(nb_samples):
            new_point = model.predict(data)
            new_point = self.scaler.inverse_transform(new_point.reshape(-1, 1))[0]
            test_predict.append(new_point)
            data = np.array([data[0][1:]])
            data = np.insert(data, row-1, new_point, 1)
        return np.array(test_predict)

    def plot(self):
        pass

    def summary(self):
        pass


if __name__ == '__main__':
    import math
    import matplotlib.pyplot as plt
    from jsc import import_data
    from sklearn.metrics import mean_squared_error

    path = '../data/data_waterlevel_final_clear.csv'
    RD = import_data.WaterlevelData(path)
    waterlevel_data = RD.load_data()
    pa = 0
    LM = LstmModel(waterlevel_data['series'][:, pa])
    lookBack = 3
    trainX, trainY, testX, testY, trainSize = LM.transform_split(lookBack, 0.999, False)
    print(testX.shape)
    modelLm = LM.build_model3([3, 30, 100, 50, 1])
    # modelLm = LM.build_model([lookBack, 100, 50, 1])
    # modelLm = LM.build_model2()
    history = LM.fit(modelLm, trainX, trainY, 5, 12)
    testPredict = LM.predict_future(modelLm, testX)
    print(testPredict.shape, waterlevel_data['series'][trainSize:, pa].shape)
    test_score = math.sqrt(mean_squared_error(
        waterlevel_data['series'][trainSize+lookBack:, pa], testPredict))
    plt.plot(waterlevel_data['time'][trainSize+lookBack:],
             waterlevel_data['series'][trainSize+lookBack:, pa], label='Real Data')
    plt.plot(waterlevel_data['time'][trainSize+lookBack:], testPredict, label='Prediction')
    plt.title(waterlevel_data['header'][pa])
    plt.xlabel('Time')
    plt.ylabel('Water Level(cm)')
    plt.show()
    '''
    # summarize history for loss
    plt.plot(LM.scaler.inverse_transform(history.history['loss']).reshape(-1, 1))
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()
    '''
    print(test_score)
    #print(waterlevel_data['series'][trainSize+lookBack:, pa], testPredict)
