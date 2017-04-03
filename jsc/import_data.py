import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from utils import cast


class WaterlevelData(object):
    def __init__(self, filename):
        """
        :param
            filename: string of path and filename
                    ex: '../data/data_waterlevel_final.csv'
        """
        datafile = Path(filename)
        if datafile.is_file():
            self.filename = filename
        else:
            raise Exception('There is no such file!')

    def load_data(self):
        """
        function to load waterlevel data from csv
        :return:
            waterlevel : dictionary
                        {'header' : column name,
                        'timestamp': datetime,
                        'series': waterlevel series from 11 water gauges}
        """
        header, time_series = self._read_data()
        time, series = self._restructure_data(time_series)
        waterlevel = {'header': header[2:],
                    'time': time,
                    'series': series}
        return waterlevel

    def _restructure_data(self, time_series):
        """
        separate datetime and the series into 2 variable
        :param
            time_series: list of date, time, and waterlevel data
        :return:
            timestamp = datetime list
            series = (numpy array float) waterlevel value at the time
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
    path = '../data/data_waterlevel_final.csv'
    RD = WaterlevelData(path)
    waterlevel_data = RD.load_data()
    print(waterlevel_data['header'])
