import csv
import numpy as np
import sys

all_data = []
v_data = []
val = []
test = []

t = sys.argv[1]

with open('./datav6_t{}/norm_data.csv'.format(t), newline='') as csvfile:
    rows = csv.reader(csvfile)
    for r in rows:
        all_data.append(r)

with open('./datav6_t{}/norm_vdata.csv'.format(t), newline='') as csvfile:
    rows = csv.reader(csvfile)
    for r in rows:
        v_data.append(r)

with open('./datav6_t{}/norm_val.csv'.format(t), newline='') as csvfile:
    rows = csv.reader(csvfile)
    for r in rows:
        val.append(r)

with open('./datav6_t{}/norm_test1.csv'.format(t), newline='') as csvfile:
    rows = csv.reader(csvfile)
    for r in rows:
        test.append(r)

all_data = all_data[1:]
v_data = v_data[1:]
val = val[1:]
test = test[1:]

all_N = len(all_data)
N = len(v_data)
n = len(val)
M = len(test)

D = len(all_data[0]) - 1

all_data_y = np.zeros((all_N,1))
all_data_x = np.zeros((all_N,D))
v_data_y = np.zeros((N,1))
v_data_x = np.zeros((N,D))
val_y = np.zeros((n,1))
val_x = np.zeros((n,D))
test_x = np.zeros((M,D))

for i in range(all_N):
    all_data_y[i] = np.array(all_data[i][0])
    all_data_x[i] = np.array(all_data[i][1:])
for i in range(N):
    v_data_y[i] = np.array(v_data[i][0])
    v_data_x[i] = np.array(v_data[i][1:])
for i in range(n):
    val_y[i] = np.array(val[i][0])
    val_x[i] = np.array(val[i][1:])
for i in range(M):
    test_x[i] = np.array(test[i])

#Write Your Code