import sys
import btk
import glob
import random
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neural_network import MLPClassifier
from collections import defaultdict
from enum import Enum
from math import exp

def get_data(nb_captor, list_of_files):
    captor = nb_captor
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
                                    acq.GetPoint('LTOE').GetValues()[event_frame,2],acq.GetPoint('RTOE').GetValues()[event_frame,2],
                                    acq.GetPoint('LHEE').GetValues()[event_frame,2],acq.GetPoint('RHEE').GetValues()[event_frame,2],
                                    acq.GetPoint('LANK').GetValues()[event_frame,2],acq.GetPoint('RANK').GetValues()[event_frame,2]])
                tmp_set = np.array([tmp_set])
                data_set = np.concatenate((data_set, tmp_set), axis=0)
                if (acq.GetEvent(i).GetLabel() == 'Foot_Off_GS' or acq.GetEvent(i).GetLabel() == 'Foot_Strike_GS'):
                    range_no_event = range(event_frame-4,event_frame-1) + range(event_frame+2,event_frame+5)
                    #print(range_no_event)
                    for j in range_no_event:
                        no_event_set = np.array(['No_Event',
                                                acq.GetPoint('LTOE').GetValues()[j,2],acq.GetPoint('RTOE').GetValues()[j,2],
                                                acq.GetPoint('LHEE').GetValues()[j,2],acq.GetPoint('RHEE').GetValues()[j,2],
                                                acq.GetPoint('LANK').GetValues()[j,2],acq.GetPoint('RANK').GetValues()[j,2]])
                        no_event_set = np.array([no_event_set])
                        data_set = np.concatenate((data_set, no_event_set), axis=0)
            except Exception as e:
                data_set = np.delete(data_set, 0, 0)
                break
    y = data_set[:,0]
    X = data_set[:,1:captor+1].astype(np.float)
    return (X,y)

def get_prediction(nb_captor,X,y,algo, files, list_of_files):
    captor = nb_captor
    sum_error_glob = 0
    sum_error_FO = 0
    sum_error_FS = 0

    pred_KNN_centroid_update = []

    if (algo == 'DT'):
        clf = tree.DecisionTreeClassifier()
    elif (algo == 'NB'):
        clf = GaussianNB()
    elif (algo == 'KNN_Centroid'):
        clf = NearestCentroid()
    elif (algo == 'KNN'):
        clf = KNeighborsClassifier(n_neighbors=3)
    elif (algo == 'MLP'):
        clf = MLPClassifier(hidden_layer_sizes=(15, 15, 15), max_iter = 500)
    clf.fit(X, y)

    for file_name in list_of_files:
        print("\n")
        X_test = np.array([np.zeros(captor)])
        array_real_event = np.array([np.zeros(2)])
        reader = btk.btkAcquisitionFileReader()
        reader.SetFilename(file_name)
        reader.Update()
        acq = reader.GetOutput()
        for i in range(100):
            try:
                event_frame = acq.GetEvent(i).GetFrame()
                tmp_array_real_event = np.array([event_frame, acq.GetEvent(i).GetLabel()])
                tmp_array_real_event = np.array([tmp_array_real_event])
                array_real_event = np.concatenate((array_real_event, tmp_array_real_event), axis=0)
            except Exception as e:
                array_real_event = np.delete(array_real_event, 0, 0)
                break
        print(file_name)
        for frame in range(1,500,1):
            try:
                tmp_X_test = np.array([acq.GetPoint('LTOE').GetValues()[frame,2],acq.GetPoint('RTOE').GetValues()[frame,2],
                                        acq.GetPoint('LHEE').GetValues()[frame,2],acq.GetPoint('RHEE').GetValues()[frame,2],
                                        acq.GetPoint('LANK').GetValues()[frame,2],acq.GetPoint('RANK').GetValues()[frame,2]])
                tmp_X_test = np.array([tmp_X_test])
                X_test = np.concatenate((X_test, tmp_X_test), axis=0)
            except Exception as e:
                X_test = np.delete(X_test, 0, 0)
                break
        if (algo == 'DT'):
            pred_decisionTree = decision_tree(X,y,X_test,clf)
        elif (algo == 'NB'):
            pred_gaussianNB = gaussian_naive_bayes(X,y,X_test,clf)
        elif (algo == 'KNN_Centroid'):
            pred_KNN_centroid = K_NN_Centroid(X,y,X_test,clf)
            res = update_label(pred_KNN_centroid, array_real_event)
            pred_KNN_centroid_update.append(res[0])
            sum_error_glob = sum_error_glob + res[1]
            sum_error_FO = sum_error_FO + res[2]
            sum_error_FS = sum_error_FS + res[3]
        elif (algo == 'KNN'):
            pred_KNN = K_NN(X,y,X_test,clf)
        elif (algo == 'MLP'):
            pred_MLP = MLP(X,y,X_test,clf)

    print(' ')
    # print('Error global: ', sum_error_glob)
    # print('Error FO: ', sum_error_FO)
    # print('Error FS: ', sum_error_FS)

    error_global, error_FO, error_FS = compute_error(sum_error_glob, sum_error_FO, sum_error_FS, files)
    # print('SCORE CP FO & FS: ', error_global) # SCORE CP FO & FS : 1209
    # print('SCORE CP FO: ', error_FO) # SCORE CP FO : 547
    # print('SCORE CP FS: ', error_FS) # SCORE CP FS : 661

    print('SCORE FD FO & FS: ', error_global) # SCORE FD FO & FS : 519
    print('SCORE FD FO: ', error_FO) # SCORE FD FO : 294
    print('SCORE FD FS: ', error_FS) # SCORE FD FS : 225

    # print('SCORE ITW FO & FS: ', error_global) # ERROR CP_FO_AND_FS : 5288
    # print('SCORE ITW FO: ', error_FO) # ERROR CP_FO : 2895
    # print('SCORE ITW FS: ', error_FS) # ERROR CP_FS : 2393
    return pred_KNN_centroid_update, error_global

def compute_error(sum_error_glob, sum_error_FO, sum_error_FS, files):
    error_global_ = 0
    error_FO_ = 0
    error_FS_ = 0
    if (files == 'CP'):
        error_global_ =  30 * exp(1) * (sum_error_glob*1.00/45)
        error_FO_ =  30 * exp(1) * (sum_error_FO*1.00/45)
        error_FS_ =  30 * exp(1) * (sum_error_FS*1.00/45)
    elif (files == 'FD'):
        error_global_ =  30 * exp(1) * float(sum_error_glob*1.00/25)
        error_FO_ =  30 * exp(1) * float(sum_error_FO*1.00/25)
        error_FS_ =  30 * exp(1) * float(sum_error_FS*1.00/25)
    elif (files == 'ITW'):
        error_global_ =  30 * exp(1) * (sum_error_glob*1.00/20)
        error_FO_ =  30 * exp(1) * (sum_error_FO*1.00/20)
        error_FS_ =  30 * exp(1) * (sum_error_FS*1.00/20)
    return error_global_, error_FO_, error_FS_

def decision_tree(X,y,X_test,clf):
    #range_no_event = range(event_frame-9,event_frame-1) + range(event_frame+2,event_frame+10)
    predictions_DT = clf.predict(X_test)
    df = pd.DataFrame(data = predictions_DT, columns = ['predictions_DT'])
    print(df.loc[df['predictions_DT'] != 'No_Event'])
    #print(df)
    return predictions_DT

def gaussian_naive_bayes(X,y,X_test,clf):
    predictions_NB = clf.predict(X_test)
    df = pd.DataFrame(data = predictions_NB, columns = ['predictions_NB'])
    print(df.loc[df['predictions_NB'] != 'No_Event'])
    #print(df)
    return predictions_NB

def K_NN_Centroid(X,y,X_test,clf):
    # Foot_Off_GS premier de l'intervalle
    # Foot_Strike_GS dernier de l'intervalle
    predictions_KNN_centroid = clf.predict(X_test)
    df = pd.DataFrame(data = predictions_KNN_centroid, columns = ['predictions_KNN_centroid'])
    #print(df.loc[df['predictions_KNN_centroid'] != 'No_Event'].to_string())
    #print(df.to_string())
    return predictions_KNN_centroid

def K_NN(X,y,X_test,clf):
    #range_no_event = range(event_frame-4,event_frame-1) + range(event_frame+2,event_frame+5)
    predictions_KNN = clf.predict(X_test)
    df = pd.DataFrame(data = predictions_KNN, columns = ['predictions_KNN'])
    #print(df.loc[df['predictions_KNN'] != 'No_Event'])
    #print(df.to_string())
    return predictions_KNN

def MLP(X,y,X_test,clf):
    #range_no_event = range(event_frame-4,event_frame-1) + range(event_frame+2,event_frame+5)
    predictions_MLP = clf.predict(X_test)
    df = pd.DataFrame(data = predictions_MLP, columns = ['predictions_MLP'])
    #print(df.loc[df['predictions_MLP'] != 'No_Event'])
    #print(df.to_string())
    return predictions_MLP

def update_label(predict_list, real_event):
    error_computed = 0
    cur_label = predict_list[0]
    index_list = []
    for index in range(len(predict_list)):
        if predict_list[index] == cur_label:
            index_list.append(index)
        if predict_list[index] != cur_label or index == len(predict_list)-1:
            size = len(index_list)
            if predict_list[index_list[0]] == 'Foot_Off_GS':
                if size > 4:
                    predict_list[index_list[0]:index_list[3]] = 'No_Event'
                    predict_list[index_list[4]:index_list[size-1]+1] = 'No_Event'
                elif size <= 4 and size > 1:
                    predict_list[index_list[0]:index_list[size-1]] = 'No_Event'
            elif predict_list[index_list[0]] == 'Foot_Strike_GS':
                if size > 4:
                    predict_list[index_list[0]:index_list[size-3]] = 'No_Event'
                    predict_list[index_list[size-2]:index_list[size-1]+1] = 'No_Event'
                elif size <= 4 and size > 1:
                    predict_list[index_list[1]:index_list[size-1]+1] = 'No_Event'
            index_list = []
            if index != len(predict_list)-1:
                index_list = []
                index_list.append(index)
                cur_label = predict_list[index]
            else:
                #df = pd.DataFrame(data = predict_list, columns = ['predict_list'])
                #print(df.loc[df['predict_list'] != 'No_Event'])
                avg_dist_FO_FS, avg_dist_FS_FO = calculate_avarage(predict_list)
                predict_list_completed = complete_data(predict_list, avg_dist_FO_FS, avg_dist_FS_FO)
                error_computed_glob, error_computed_FO, error_computed_FS = calcul_error(predict_list_completed, real_event)
                print('Error file: ', error_computed_glob)
                break
    df = pd.DataFrame(data = predict_list_completed, columns = ['predict_list_completed'])
    print(df.loc[df['predict_list_completed'] != 'No_Event'])
    return predict_list_completed, error_computed_glob, error_computed_FO, error_computed_FS

def calculate_avarage(list_predictions):
    count_FS_FO = 0
    count_FO_FS = 0
    count_FS_FS = 0
    count_FO_FO = 0
    save_event = ''
    save_ind = 0
    dist_FO_FS = 0
    dist_FS_FO = 0
    dist_FO_FO = []
    dist_FS_FS = []

    for i in range(len(list_predictions)):
        current_event = list_predictions[i]
        if (current_event == 'Foot_Off_GS' and save_event == ''):
            save_event = 'Foot_Off_GS'
            save_ind = i
        elif (current_event == 'Foot_Strike_GS' and save_event == ''):
            save_event = 'Foot_Strike_GS'
            save_ind = i
        elif (current_event == 'Foot_Strike_GS' and save_event == 'Foot_Off_GS'):
            count_FO_FS += 1
            dist_FO_FS = dist_FO_FS + (i - save_ind)
            save_event = 'Foot_Strike_GS'
            save_ind = i
        elif (current_event == 'Foot_Off_GS' and save_event == 'Foot_Strike_GS'):
            count_FS_FO += 1
            dist_FS_FO = dist_FS_FO + (i - save_ind)
            save_event = 'Foot_Off_GS'
            save_ind = i
        elif (current_event == 'Foot_Off_GS' and save_event == 'Foot_Off_GS'):
            dist_FO_FO.append(i - save_ind)
            save_ind = i
            count_FO_FO += 1
        elif (current_event == 'Foot_Strike_GS' and save_event == 'Foot_Strike_GS'):
            dist_FS_FS.append(i - save_ind)
            save_ind = i
            count_FS_FS += 1

    #FIXME: when not enough prediction because big range so FO_FO FS_FS can appear

    if (count_FS_FO == 0 and count_FO_FS == 0):
        if (count_FO_FO == 0 and count_FS_FS == 0):
            avg_dist_FO_FS = 0
            avg_dist_FS_FO = 0
        else:
            if (count_FO_FO == 0):
                avg_dist_FO_FS = ((sum(dist_FS_FS)/len(dist_FS_FS))*2)/3
                avg_dist_FS_FO = (sum(dist_FS_FS)/len(dist_FS_FS))/3
            if (count_FS_FS == 0):
                avg_dist_FO_FS = ((sum(dist_FO_FO)/len(dist_FO_FO))*2)/3
                avg_dist_FS_FO = (sum(dist_FO_FO)/len(dist_FO_FO))/3
    elif (count_FS_FO != 0 and count_FO_FS == 0):
        avg_dist_FS_FO = dist_FS_FO/count_FS_FO
        if (count_FO_FO != 0):
            avg_dist_FO_FS = (sum(dist_FO_FO)/len(dist_FO_FO)) - avg_dist_FS_FO
        else:
            avg_dist_FO_FS = 2*avg_dist_FS_FO
    elif (count_FS_FO == 0 and count_FO_FS != 0):
        avg_dist_FO_FS = dist_FO_FS/count_FO_FS
        if (count_FS_FS != 0):
            avg_dist_FS_FO = (sum(dist_FS_FS)/len(dist_FS_FS)) - avg_dist_FO_FS
        else:
            avg_dist_FS_FO = avg_dist_FO_FS/2
    else:
        avg_dist_FS_FO = dist_FS_FO/count_FS_FO
        avg_dist_FO_FS = dist_FO_FS/count_FO_FS

    #print('avg FO-FS: ',avg_dist_FO_FS)
    #print('avg FS-FO: ',avg_dist_FS_FO)
    return (avg_dist_FO_FS, avg_dist_FS_FO)

def complete_data(list_predictions, FO_FS, FS_FO):
    current_label = ''
    save_label = ''
    save_ind = 0
    for i in range(len(list_predictions)):
        current_label = list_predictions[i]
        if (current_label == 'Foot_Off_GS' and (save_label == '' or save_label == 'Foot_Strike_GS')):
            save_label = 'Foot_Off_GS'
            save_ind = i
        elif (current_label == 'Foot_Strike_GS' and (save_label == '' or save_label == 'Foot_Off_GS')):
            save_label = 'Foot_Strike_GS'
            save_ind = i
        elif (current_label == 'Foot_Off_GS' and save_label == 'Foot_Off_GS'):
            list_predictions[i-FS_FO] = 'Foot_Strike_GS'
        elif (current_label == 'Foot_Strike_GS' and save_label == 'Foot_Strike_GS'):
            list_predictions[i-FO_FS] = 'Foot_Off_GS'
    return list_predictions


def calcul_error(predict_list, tab_real_event):
    number_frame_error = 0
    number_frame_error_FO = 0
    number_frame_error_FS = 0

    array_pred_list = np.array([np.zeros(2)])
    for i in range(len(predict_list)):
        tmp_array_pred_list = np.array([i, predict_list[i]])
        tmp_array_pred_list = np.array([tmp_array_pred_list])
        if (predict_list[i] != 'No_Event'):
            array_pred_list = np.concatenate((array_pred_list, tmp_array_pred_list), axis=0)
    array_pred_list = np.delete(array_pred_list, 0, 0)

    for real_evt in range(len(tab_real_event)):
        min_dist = 1000
        for pred_evt in range(len(array_pred_list)):
            if (tab_real_event[real_evt,1] == array_pred_list[pred_evt,1]):
                if (abs(tab_real_event[real_evt,0].astype(np.int) - array_pred_list[pred_evt,0].astype(np.int)) < min_dist):
                    min_dist = abs(tab_real_event[real_evt,0].astype(np.int)-array_pred_list[pred_evt,0].astype(np.int))

        if (tab_real_event[real_evt,1] == 'Foot_Off_GS'):
            number_frame_error_FO = number_frame_error_FO + min_dist
        elif (tab_real_event[real_evt,1] == 'Foot_Strike_GS'):
            number_frame_error_FS = number_frame_error_FS + min_dist

        number_frame_error = number_frame_error + min_dist
    return number_frame_error, number_frame_error_FO, number_frame_error_FS

def main():
    nb_capt = 6
    files = 'FD'
    list_of_files = glob.glob('./Sofamehack2019/Sub_DB_Checked/'+files+'/*.c3d')
    [X,y] = get_data(nb_capt, list_of_files)
    pred, error = get_prediction(nb_capt,X,y,'KNN_Centroid', files, list_of_files)

main()
