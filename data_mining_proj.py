import sys
import btk
import glob
import random
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB


def get_data_set():
    list_of_files = glob.glob('./Sofamehack2019/Sub_DB_Checked/CP/*.c3d')
    captor = 6
    data_set = np.array([np.zeros(captor+1)])

    for file_name in list_of_files:
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(file_name)
        reader.Update()
        acq = reader.GetOutput()

        for i in range(100):
            try:
                event_frame = acq.GetEvent(i).GetFrame()
                tmp_set = np.array([acq.GetEvent(i).GetLabel(),
                                    acq.GetPoint('LTOE').GetValues()[event_frame,2],
                                    acq.GetPoint('RTOE').GetValues()[event_frame,2],
                                    acq.GetPoint('RANK').GetValues()[event_frame,2],
                                    acq.GetPoint('LANK').GetValues()[event_frame,2],
                                    acq.GetPoint('RHEE').GetValues()[event_frame,2],
                                    acq.GetPoint('LHEE').GetValues()[event_frame,2]])
                tmp_set = np.array([tmp_set])
                data_set = np.concatenate((data_set, tmp_set), axis=0)
                if (acq.GetEvent(i).GetLabel() == 'Foot_Off_GS' or acq.GetEvent(i).GetLabel() == 'Foot_Strike_GS'):
                    for j in [event_frame-5, event_frame-2, event_frame+2, event_frame+5]:
                        no_event_set = np.array(['No_Event',
                                                acq.GetPoint('LTOE').GetValues()[j,2],
                                                acq.GetPoint('RTOE').GetValues()[j,2],
                                                acq.GetPoint('RANK').GetValues()[j,2],
                                                acq.GetPoint('LANK').GetValues()[j,2],
                                                acq.GetPoint('RHEE').GetValues()[j,2],
                                                acq.GetPoint('LHEE').GetValues()[j,2]])
                        no_event_set = np.array([no_event_set])
                        data_set = np.concatenate((data_set, no_event_set), axis=0)
            except Exception as e:
                data_set = np.delete(data_set, 0, 0)
                break
    return data_set

def split_data():
    data_set = get_data_set()
    #np.random.shuffle(data_set)
    train_set = data_set[0:(data_set.shape[0]*2/3),:]
    test_set = data_set[(data_set.shape[0]*2/3):data_set.shape[0],:]
    return (train_set, test_set)

def test_set():
    captor = 6
    #train_set = split_data()[0]
    train_set = get_data_set()
    test_set_tmp = split_data()[1]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_set[:,1:captor], train_set[:,0])

    #list_of_files = glob.glob('./Sofamehack2019/Sub_DB_Checked/CP/*.c3d')
    list_of_files= ['./Sofamehack2019/Sub_DB_Checked/CP/CP_GMFCS1_01916_20130128_18.c3d']

    for file_name in list_of_files:
        test_set = np.array([np.zeros(captor)])
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(file_name)
        reader.Update()
        acq = reader.GetOutput()
        print(file_name)
        for i in range(0,500,1):
            try:
                tmp_set = np.array([acq.GetPoint('LTOE').GetValues()[i,2],
                                    acq.GetPoint('RTOE').GetValues()[i,2],
                                    acq.GetPoint('RANK').GetValues()[i,2],
                                    acq.GetPoint('LANK').GetValues()[i,2],
                                    acq.GetPoint('RHEE').GetValues()[i,2],
                                    acq.GetPoint('LHEE').GetValues()[i,2]])
                tmp_set = np.array([tmp_set])
                test_set = np.concatenate((test_set, tmp_set), axis=0)
            except Exception as e:
                test_set = np.delete(test_set, 0, 0)
                break
        res = clf.predict(test_set[:,1:captor])

        df = pd.DataFrame(data = res, columns = ['predictions'])
        print(df.loc[df['predictions'] != 'No_Event'])
        #print(df)
    return test_set


# ---- RUN ---- #
print(test_set())
