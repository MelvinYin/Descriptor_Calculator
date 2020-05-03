
import pickle

class Test:
    VERSION = 3

    def __init__(self):
        self.x = 1
        self.version = self.__class__.VERSION
        self.my_version = 2

    def print_this(self):
        if self.version != self.__class__.VERSION:
            print("ERROR")
        if self.version != self.my_version:
            print("THIS IS WRONG")
        print(self.x)


from bokeh.palettes import Category10

x = Category10
print(x)

# a = Test()
#
# print(a.version)
# a.print_this()
#
# with open("test.pkl", 'wb') as file:
#     pickle.dump(a, file, -1)

with open("test.pkl", 'rb') as file:
    b = pickle.load(file)

print(b.version)
b.print_this()
