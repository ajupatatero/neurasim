from util.performance.timer import *



tt = Timer()


tt.record('tt1')


tt.record('tt2')

name, time, time_dif, percentage = tt.close(save=True)

print(name)
print(time)
print(time_dif)