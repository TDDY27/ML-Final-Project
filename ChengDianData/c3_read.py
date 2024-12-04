import csv
import numpy as np

data = []
test = []

with open('./c3_empty_raw_data_full_column', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for r in rows:
        data.append(r)

with open('./c3_empty_raw_test1_full_column', newline='') as csvfile:
    rows = csv.reader(csvfile)
    for r in rows:
        test.append(r)

data = data[1:]
test = test[1:]

N = len(data)
M = len(test)

D = len(data[0]) - 1

data_y = np.zeros((N,1))
data_x = np.zeros((N,D))
test_x = np.zeros((M,D))

for i in range(N):
    data_y[i] = np.array(data[i][0])
    data_x[i] = np.array(data[i][1:])
for i in range(M):
    test_x[i] = np.array(test[i])

#Write Your Code