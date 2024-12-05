import sys
import subprocess

a = int(sys.argv[1])
b = int(sys.argv[2])

for t in range(a,b):
    subprocess.call(["rm", "-rf","./datav5_t{}".format(t)])
    subprocess.call(["mkdir", "./datav5_t{}".format(t)])

subprocess.call(["python3", "process_data_v5.py", "{}".format(a), "{}".format(b)])
subprocess.call(["python3", "reduce_data_v5.py", "{}".format(a), "{}".format(b)])
subprocess.call(["python3", "process_val_v5.py", "{}".format(a), "{}".format(b)])