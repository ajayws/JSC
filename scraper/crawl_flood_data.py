import urllib.request
import json
import csv


def get_flood_data(url):
    data = json.load(urllib.request.urlopen(url))
    if data["features"] is None:
        return None
    else:
        lon = data["features"][0]["geometry"]["coordinates"][0]
        lat = data["features"][0]["geometry"]["coordinates"][1]
        datetime = str(data["features"][0]["properties"]["created_at"])
        return datetime, lon, lat


def main():
    url = "https://petajakarta.org/banjir/data/api/v2/reports/confirmed/"
    start_index = 1
    column_name = ["datetime", "longitude", "latitude"]
    with open('../data/flood_data.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(column_name)
        while True:
            row = [None] * 3
            row = get_flood_data(url + str(start_index))
            if row is None:
                break
            writer.writerow(row)
            print(start_index)
            start_index += 1


if __name__ == '__main__':
    main()
