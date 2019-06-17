import sys
import btk

reader = btk.btkAcquisitionFileReader() # build a btk reader object
reader.SetFilename("CP_GMFCS1_01916_20130128_18.c3d") # set a filename to the reader
reader.Update()
acq = reader.GetOutput() # acq is the btk aquisition object

freq = acq.GetPointFrequency() # give the point frequency
print('freq : ', freq)
n_frames = acq.GetPointFrameNumber() # give the number of frames
print('n_frames : ', n_frames)
first_frame = acq.GetFirstFrame()
print('first_frame ', first_frame)
