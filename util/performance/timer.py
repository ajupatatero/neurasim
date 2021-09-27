import torch
import csv

import pdb

class Timer:
    def __init__(self, out):
        self._outdir = out
        self._time_points = []
        self._name_points = []

        self._single_interval = False
        self._time_interval = []
        self._name_interval = []

        self.record(point_name='INIT')

    def record(self, point_name='unnamed point'):
        
        #TODO: something to calculate accumulative for plots, difuse, etc. given the name, or an aditional parameter as a identifier
        #same with diference between diferent points not consecutive


        #OR BETTER -> do recor (id = 11) -> i llavors que afegis el append en el mateix row que el id donat !!! aixi fer tot els totals, mean, etc

        self._time_points.append(torch.cuda.Event(enable_timing=True))
        self._time_points[-1].record()
        self._name_points.append(point_name)
    
    def add_single_interval(self, interval_time, interval_name='unnamed interval'):
        self._single_interval = True
        self._time_interval.append(interval_time)
        self._name_interval.append(interval_name)

    def clean(self):
        self.__init__()

    def close(self, save=False):

        self.record(point_name='END')
        torch.cuda.synchronize()  # Wait for the events to be recorded! 

        time = [0]
        time_dif = [0]
        for i, event in enumerate(self._time_points):

            if i == len(self._time_points)-1:
                break
            else:
                elapsed_time_ms = event.elapsed_time(self._time_points[i+1])
                time_dif.append(elapsed_time_ms)
                time.append(time[-1]+elapsed_time_ms)

        percentage = []
        for i, dif in enumerate(time_dif):
            percentage.append(round((dif/time[-1])*100, 2))

        if self._single_interval:
            for j, inter in enumerate(self._time_interval):
                self._name_points.append(self._name_interval[j])
                time_dif.append(inter)
                time.append(0)
                percentage.append(0)
        

        if save:
            with open(f'{self._outdir}performance_results.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([ 'id', 'POINT_NAME', 'TIME', 'TIME_DIFFERENCE', '% of time'])

                for i in range(len(time)):
                    writer.writerow([ i, self._name_points[i], time[i], time_dif[i], percentage[i]])

        return self._name_points, time, time_dif, percentage
        