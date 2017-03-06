import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from utils import cast


class RainfallData(object):
    def __init__(self, filename):
        """
        :param
            filename: string of path and filename
                    ex: '../data/data_rainfall_final.csv'
        """
        datafile = Path(filename)
        if datafile.is_file():
            self.filename = filename
        else:
            raise Exception('There is no such file!')

    def load_data(self):
        """
        function to load rainfall data from csv
        :return:
            rainfall : dictionary
                        {'header' : column name,
                        'timestamp': datetime,
                        'series': rainfall series from 11 water gauges}
        """
        header, time_series = self._read_data()
        time, series = self._restructure_data(time_series)
        rainfall = {'header': header,
                    'time': time,
                    'series': series}
        return rainfall

    def _restructure_data(self, time_series):
        """
        seperate datetime and the series into 2 variable
        :param
            time_series: list of date, time, and rainfall data
        :return:
            timestamp = datetime list
            series = (numpy array float) rainfall value at the time
        """
        time = []
        series = []
        for row in time_series:
            time.append(datetime.strptime(row[0] + ' ' + row[1],
                        '%m/%d/%Y %H:%M'))
            series.append([cast.float_or_nan(val) for val in row[2:]])
        series = np.array(series)
        return time, series

    def _read_data(self):
        """
        IO reader from csv file
        :return:
            header : list (column name)
            time_series : list array (time series data)
        """
        with open(self.filename, 'r') as csv_file:
            data_reader = csv.reader(csv_file, delimiter=',')
            header = next(data_reader)
            time_series = [row for row in data_reader]
        return header, time_series

if __name__ == '__main__':
    path = '../data/data_rainfall_final.csv'
    RD = RainfallData(path)
    rainfall_data = RD.load_data()
    print(rainfall_data['series'][:, 0])
