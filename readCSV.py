import csv

Dict = []
with open("archive/english.csv", "r") as data:
    for line in csv.DictReader(data):
        Dict.append(line)

    print(Dict)



