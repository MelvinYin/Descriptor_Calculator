
import os
from config import paths

filename = os.path.join(paths.ROOT, "part-r-00000 (10)")
counter = 0
with open(filename, 'r') as file:
    for line in file:
        a, b = line.split("\t")
        print(a)
        print(b)
        print("")
        counter += int(a) * int(b)
print(counter)