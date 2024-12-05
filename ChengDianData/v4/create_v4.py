import sys
import subprocess

a = int(sys.argv[1])
b = int(sys.argv[2])

for t in range(a,b):
    subprocess.call(["rm", "-rf","./datav4_t{}".format(t)])
    subprocess.call(["mkdir", "./datav4_t{}".format(t)])

subprocess.call(["python3", "process_data_v4.py", "{}".format(a), "{}".format(b)])
subprocess.call(["python3", "process_val_v4.py", "{}".format(a), "{}".format(b)])