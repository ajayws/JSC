from statsmodels.tsa.arima_model import ARIMA


class ArimaModel(object):
    def __init__(self, series):
        self.series = series

    def transform_split(self, train_composition=0.7):
        """
        Split series into train and test data
        :param
            train_composition : (float > 0) percentage or index
        :return :
            training : Array of training data
            test : Array of testing data
            train_size : training size
        """
        if train_composition > 1 or train_composition < 0:
            raise Exception('Train composition must be between 0-1')
        train_size = int(len(self.series) * train_composition)
        train = self.series[0:train_size]
        test = self.series[train_size:]
        return train, test, train_size

    @staticmethod
    def predict(train, test, p, d, q):
        """
        Use this function to predict only one single point(value at t+1)
        based on prior true history
        :param
            train : Array of training data
            test : Array of testing data
            p : parameter p in ARIMA(p, d, q)
            d : parameter d in ARIMA(p, d, q)
            q : parameter q in ARIMA(p, d, q)
        :return:
            test_predict : Array testing data
        """
        history = list(train)
        test_predict = []
        for i in range(len(test)):
            model = ARIMA(history, order=(p, d, q))
            model = model.fit(disp=0)
            output = model.forecast()
            test_predict.append(output[0])
            history.append(test[i])
            print(i)
        return test_predict

    @staticmethod
    def predict_future(train, test, p, d, q):
        """
        Use this function to predict multiple value in the future
        at time t+1, t+2, t+3, etc
        :param
            train : Array - training data
            test : Array testing data
            p : parameter p in ARIMA(p, d, q)
            d : parameter d in ARIMA(p, d, q)
            q : parameter q in ARIMA(p, d, q)
        :return:
            test_predict : Array testing data
        """
        history = list(train)
        test_predict = []
        for i in range(len(test)):
            model = ARIMA(history, order=(p, d, q))
            model = model.fit(disp=0)
            output = model.forecast()
            test_predict.append(output[0])
            history.append(output[0])
            print(i)
        return test_predict


if __name__ == '__main__':
    import math
    import matplotlib.pyplot as plt
    from jsc import import_data
    from sklearn.metrics import mean_squared_error

    path = '../data/data_rainfall_final_clear.csv'
    RD = import_data.RainfallData(path)
    rainfall_data = RD.load_data()
    pa = 2
    AM = ArimaModel(rainfall_data['series'][:, pa])
    trainD, testD, trainSize = AM.transform_split(0.8)
    testPredict = AM.predict(trainD, testD, 3, 0, 0)
    print(len(testPredict))
    print(len(rainfall_data['series'][trainSize:, pa]))
    test_score = math.sqrt(mean_squared_error(
        rainfall_data['series'][trainSize:, pa], testPredict))
    plt.plot(rainfall_data['time'][trainSize:],
             rainfall_data['series'][trainSize:, pa], label='Real Data')
    plt.plot(rainfall_data['time'][trainSize:], testPredict, label='Prediction')
    plt.show()
    plt.title('Pintu Air Manggarai')
    plt.xlabel('Time')
    plt.ylabel('Water Level(cm)')
    print(testPredict[1:10])
    print(test_score)
