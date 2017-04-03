import requests
import csv
from datetime import timedelta, date
from BeautifulSoup import BeautifulSoup

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        d = start_date + timedelta(n)
        yield d.strftime("%d-%m-%Y")

def getRainfallTable(url, params, headers):
    r = requests.post(url, params, headers)
    soup = BeautifulSoup(r.content)
    table = soup.find("div", {"id":"table2", "class":"table2"})
    head = table.table.thead
    body = table.table.tbody
    thList = head.findAll("th")
    trList = body.findAll("tr")
    data = []
    col = []
    if len(thList) > 2:
        for i in xrange(1,len(thList)):
            col.append(thList[i].text.strip())
        data.append(col)
        for tr in trList:
            row = []
            tdList = tr.findAll("td")
            for td in tdList:
                #append to list
                row.append(td.text.strip())
            data.append(row)
    data = map(list, zip(*data))
    return data

def main():
    headers = {
        'Host': 'bpbd.jakarta.go.id',
        'Connection': 'keep-alive',
        'Content-Length': 15,
        'Origin': 'http://bpbd.jakarta.go.id',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'http://bpbd.jakarta.go.id/index.php/waterlevel/',
    }
    url = 'http://bpbd.jakarta.go.id/index.php/waterlevel/'
    start_date = date(2014, 5, 19)
    end_date = date(2016, 9, 30)
    #column headers
    column_name = ['Date', 'Time', 'Bendung Katulampa', 'Pos Depok', 'PA Manggarai', 'PA Karet', 
                'Pos Krukut Hulu', 'PA Pesanggrahan', 'Pos Angke Hulu', 'Waduk Pluit', 'Pasar Ikan',
                'Pos Cipinang Hulu', 'Pos Sunter Hulu', 'PA Pulo Gadung']
    time_row = ['0%i:00' % (i,) for i in xrange(10)] + ['%i:00' % (i,) for i in xrange(10,24)]
    with open('rainfall_data2.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(column_name)
        for d in daterange(start_date, end_date):
            params = {'date': d}
            table = getRainfallTable(url, params, headers)
            print
            print len(table)
            for i in xrange(len(table)):
                row = [d] + table[i]
                writer.writerow(row)


if __name__ == '__main__':
    main()