import numpy as np
import matplotlib.pyplot as plt

f = open("log.txt", "r")
log = f.read()
log = log.split(" ")

def filter_digits(s):
	return str.isdigit(s) or s == '.'

res = []
for l in log:
	numeric_filter = filter(filter_digits , l)
	numeric_string = "".join(numeric_filter)
	if numeric_string != "" and float(numeric_string) < 1 and float(numeric_string) > 0:
		res.append(numeric_string)

log = np.array(res).astype(np.float)
print(log)

log = log.reshape(-1,6)
test_accs = log[:,-1]
plt.bar(range(32),test_accs)
plt.show()
print(test_accs.mean())
# test_accs = 

