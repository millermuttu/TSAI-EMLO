from pathlib import Path
import csv

datafile = Path("Session1/data.dvc")
try:
    my_abs_path = datafile.resolve(strict=True)
except FileNotFoundError:
    print("data file found")
else:
    print("data file not found!!")


model = Path("Session1/model_pytorch.h5.dvc")
try:
    my_abs_path = datafile.resolve(strict=True)
except FileNotFoundError:
    print("data file found")
else:
    print("data file not found!!")


with open('metrics.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    accracy = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            temp = row[3]
            if temp>accracy:
                accracy = temp
    csv_file.close()
    if accracy>70:
        print("Accuracy is achived")
    else:
        print("bad accuracy")