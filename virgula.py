import csv
from csv import reader


list_v1 = []
nr=1
with open('train.csv','r') as csv_file:
   csv_reader = csv.reader(csv_file, delimiter=',')
   for row in csv_reader:

       if (len(row)-1 == 2):
           print(row)
           
