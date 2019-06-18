import sys
import numpy as np
import btk
import glob

reader = btk.btkAcquisitionFileReader() # build a btk reader object
reader.SetFilename("CP_GMFCS1_01916_20130128_18.c3d") # set a filename to the reader
reader.Update()
acq = reader.GetOutput() # acq is the btk aquisition object

list_of_files = glob.glob('*.c3d')
for file_name in list_of_files:
    print(file_name)

def get_train_set():
    captor = 2
    train_set = np.array([np.zeros(captor+1)])
    for i in range(100):
        try:
            event_frame = acq.GetEvent(i).GetFrame()
            tmp_set = np.array([acq.GetEvent(i).GetLabel(),
                                acq.GetPoint('LHEE').GetValues()[event_frame,2],
                                acq.GetPoint('RHEE').GetValues()[event_frame,2]])
            tmp_set = np.array([tmp_set])
            train_set = np.concatenate((train_set, tmp_set), axis=0)
        except Exception as e:
            train_set = np.delete(train_set, 0, 0)
            print(train_set.shape)
            return train_set

print(get_train_set())


# for i in range(20):
#     data_FrameRef = np.array([  acq.GetPoint('LHEE').GetValues()[i,2],
#                                 acq.GetPoint('RHEE').GetValues()[i,2]])
#     print(data_FrameRef)
