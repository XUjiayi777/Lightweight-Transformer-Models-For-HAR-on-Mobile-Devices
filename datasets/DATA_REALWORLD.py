#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Uncomment if running on googlecolab 
# !pip install hickle
# from google.colab import drive
# drive.mount('/content/drive/')
# %cd drive/MyDrive/PerCom2021-FL-master/


# In[ ]:


import numpy as np
import os
import pandas as pd
import zipfile
from sklearn.preprocessing import StandardScaler
import hickle as hkl 
import requests 
import urllib.request
from scipy import signal
import tensorflow as tf
import resampy
np.random.seed(0)


# In[ ]:


# fomating data to adjust for dataset sensor errors
def formatData(data,dim):
    remainders = data.shape[0]%dim
    max_index = data.shape[0] - remainders
    data = data[:max_index,:]
    new = np.reshape(data, (-1, 128,3))
    return new

# segment data into windows
def segmentData(accData,time_step,step):
#     print(accData.shape)
    segmentAccData = list()
    for i in range(0, accData.shape[0] - time_step,step):
        segmentAccData.append(accData[i:i+time_step,:])
    return np.asarray(segmentAccData)

# load a single file as a numpy array
def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=0,usecols=[2,3,4])
    return dataframe.values
 
# load a list of files, such as x, y, z data for a given variable
def load_group(filenames, filepath='',trainOrEval=0):
    loaded = list()
    for name in filenames:
        data = load_file(filepath + name)
        data = np.asarray(data)
#         print(data.shape)
#         data = segmentData(data,128,64)
        data = np.asarray(data)
        loaded.append(data)
    return loaded

# check if file exist
def isReadableFile(file_path, file_name,flag):
    full_path = file_path + "/" + file_name
    try:
        if not os.path.exists(file_path):
            return False
        elif not os.path.isfile(full_path):
            return False
        elif not os.access(full_path, os.R_OK):
            return False
        else:
            return True
    except IOError as ex:
        print ("I/O error({0}): {1}".format(ex.errno, ex.strerror))
    except Error as ex:
        print ("Error({0}): {1}".format(ex.errno, ex.strerror))
    return False


# stairs down 0 
# stairs Up   1
# jumping     2
# lying       3
# standing    4 
# sitting     5
# running/jogging 6
# Walking     7

# load a dataset group
def load_dataset(group, datasetName='',activity='',orientation='',trainOrEval=0,client = 0):
    filepath = 'dataset/'+datasetName +'/'+ group + '/'
    filenames = list()
    if(isReadableFile('dataset/'+datasetName +'/'+ group, str(client)+'/'+group+'_'+activity+'_'+orientation+'.csv',0)):
        filenames += [str(client)+'/'+group+'_'+activity+'_'+orientation+'.csv']
    if(isReadableFile('dataset/'+datasetName +'/'+ group, str(client)+'/'+group+'_'+activity+'_2_'+orientation+'.csv',1)):
        filenames += [str(client)+'/'+group+'_'+activity+'_2_'+orientation+'.csv']
    if(isReadableFile('dataset/'+datasetName +'/'+ group, str(client)+'/'+group+'_'+activity+'_3_'+orientation+'.csv',1)):
        filenames += [str(client)+'/'+group+'_'+activity+'_3_'+orientation+'.csv']
    X = load_group(filenames, filepath,trainOrEval)
    return X

# download function for datasets
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


# In[ ]:


# definign activities and orientations of REALWORLD dataset
activities = ['climbingdown','climbingup','jumping','lying','running','sitting','standing','walking'] 
# activities = ['standing','sitting','walking','climbingup','climbingdown'] 

orientations = ['chest','forearm','head','shin','thigh','upperarm','waist']
# orientations = ['waist']


# In[ ]:


orientationKeyMap = dict() 
for index,value in enumerate(orientations):
    orientationKeyMap[value] = index


# In[ ]:


# download and unzipping dataset
os.makedirs('dataset',exist_ok=True)
print("downloading...")            
data_directory = os.path.abspath("dataset/download/realworld2016_dataset.zip")
if not os.path.exists(data_directory):
    download_url("http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip",data_directory)
    print("download done")
else:
    print("dataset already downloaded")
    
data_directory2 = os.path.abspath("dataset/realworld2016_dataset")
if not os.path.exists(data_directory2): 
    print("extracting data")
    with zipfile.ZipFile(data_directory, 'r') as zip_ref:
        zip_ref.extractall(os.path.abspath(data_directory2))
    print("data extracted in " + data_directory2)
else:
    print("Data already extracted in " + data_directory2)


# In[ ]:


# # unzipping REALWORLD dataset
for id in range(1,16):
    id = str(id)
    for activity in activities:
        for sensor in ["acc","gyr","mag"]:
            dirName = sensor
            if(sensor == "gyr"):
                dirName = "Gyroscope"
            elif (sensor=="mag"):
                dirName = "MagneticField"
            with zipfile.ZipFile('dataset/realworld2016_dataset/proband'+id+'/data/'+sensor+'_'+str(activity)+'_csv.zip', 'r') as zip_ref:
                os.makedirs('dataset/REALWORLD/'+dirName+'/'+id, exist_ok=True)
                zip_ref.extractall('dataset/REALWORLD/'+dirName+'/'+id)

            for i in range (1,4):
                if os.path.exists('dataset/REALWORLD/'+dirName+'/'+id+'/'+sensor+'_'+str(activity)+'_'+str(i)+'_csv.zip'):
                    with zipfile.ZipFile('dataset/REALWORLD/'+dirName+'/'+id+'/'+sensor+'_'+str(activity)+'_'+str(i)+'_csv.zip', 'r') as zip_ref:
                        zip_ref.extractall('dataset/REALWORLD/'+dirName+'/'+id)
                    os.remove('dataset/REALWORLD/'+dirName+'/'+id+'/'+sensor+'_'+str(activity)+'_'+str(i)+'_csv.zip') 
            if os.path.exists('dataset/REALWORLD/'+dirName+'/'+id+'/'+sensor+'_'+str(activity)+'_csv.zip'):
                with zipfile.ZipFile('dataset/REALWORLD/'+dirName+'/'+id+'/'+sensor+'_'+str(activity)+'_csv.zip', 'r') as zip_ref:
                    zip_ref.extractall('dataset/REALWORLD/'+dirName+'/'+id)
                os.remove('dataset/REALWORLD/'+dirName+'/'+id+'/'+sensor+'_'+str(activity)+'_csv.zip') 


# In[ ]:


def downSampleLowPass(toDownSampleData,factor):
    accX = signal.decimate(toDownSampleData[:,0],factor)
    accY = signal.decimate(toDownSampleData[:,1],factor)
    accZ = signal.decimate(toDownSampleData[:,2],factor)
    return np.dstack((accX,accY,accZ)).squeeze()


# In[ ]:


# Reading and processing all data
clientsOrientation = []

nbAnomalies = 0 
clientsAccDataByOrientation = []
clientsGyroDataByOrientation = []
clientMagDataByOrientation = []
clientsLabelByOrientation = []

for orientation in orientations:
    
    xAccListClient = list()
    xGyrListClient = list()
    xMagListClient = list()  
    yListClient = list()
    
    for k in range(1, 16):
        xAccList = list()
        xGyrList = list()
        xMagList = list() 
        yList = list()
        startingIndex = 0
        
        clientOrientation = []
        
        for activity in activities:
            tempAcc = load_dataset('acc', 'REALWORLD', activity, orientation, 0, k)
            tempGyro = load_dataset('Gyroscope', 'REALWORLD', activity, orientation, 0, k)
            tempMag = load_dataset('MagneticField', 'REALWORLD', activity, orientation, 0, k)  # Load magnetometer data
            orientationLength = 0 
            for i in range(0, len(tempAcc)):  
                accDataLength = len(tempAcc[i])
                gyroDataLength = len(tempGyro[i])
                magDataLength = len(tempMag[i])
                
                # Compare the lengths of accelerometer, gyroscope, and magnetometer data
                differenceAccGyro = accDataLength - gyroDataLength
                differenceAccMag = accDataLength - magDataLength
                differenceGyroAcc = gyroDataLength - accDataLength
                differenceGyroMag = gyroDataLength - magDataLength
                
                differenceAbsAccGyro = abs(differenceAccGyro)
                differenceAbsAccMag = abs(differenceAccMag)
                differenceAbsGyroMag = abs(differenceGyroMag)
                
                # Handle misalignment between accelerometer and gyroscope data
                if differenceGyroAcc > 1000:
                    print("Client Number "+str(k) +" Activity : "+str(activity) + " Orientation :"+str(orientation))
                    print("Disalignment of Acc and Gyro data: " + str(differenceAbsAccGyro) + " found")
                    print("Acc data: "+str(accDataLength))
                    print("Gyro data: "+str(gyroDataLength))
                    tempGyro[i] = resampy.resample(tempGyro[i], gyroDataLength, accDataLength, axis=0)

                # Handle misalignment between accelerometer and magnetometer data
                if differenceAbsAccMag > 1000:
                    print("Client Number "+str(k) +" Activity : "+str(activity) + " Orientation :"+str(orientation))
                    print("Disalignment of Acc and Mag data: " + str(differenceAbsAccMag) + " found")
                    print("Acc data: "+str(accDataLength))
                    print("Mag data: "+str(magDataLength))
                    tempMag[i] = resampy.resample(tempMag[i], magDataLength, accDataLength, axis=0)

                # Handle misalignment between gyroscope and magnetometer data
                if differenceAbsGyroMag > 1000:
                    print("Client Number "+str(k) +" Activity : "+str(activity) + " Orientation :"+str(orientation))
                    print("Disalignment of Gyro and Mag data: " + str(differenceAbsGyroMag) + " found")
                    print("Gyro data: "+str(gyroDataLength))
                    print("Mag data: "+str(magDataLength))
                    tempMag[i] = resampy.resample(tempMag[i], magDataLength, gyroDataLength, axis=0)

                # Segment the data (acc, gyro, mag)
                tempAcc[i] = segmentData(tempAcc[i], 128, 64)
                tempGyro[i] = segmentData(tempGyro[i], 128, 64)
                tempMag[i] = segmentData(tempMag[i], 128, 64)

                accDataLength = len(tempAcc[i])
                gyroDataLength = len(tempGyro[i])
                magDataLength = len(tempMag[i])

                difference = accDataLength - gyroDataLength
                differenceAbs = abs(difference)
                
                # If accelerometer, gyroscope, and magnetometer data lengths are compatible, proceed
                if abs(accDataLength - gyroDataLength) < 21 and abs(accDataLength - magDataLength) < 21 and abs(gyroDataLength - magDataLength) < 21:
                    toAddShape = 0
                    if accDataLength >= gyroDataLength and accDataLength >= magDataLength:
                        # If accelerometer is the longest, trim gyroscope and magnetometer data to match its length
                        maxIndex = accDataLength - max(abs(accDataLength - gyroDataLength), abs(accDataLength - magDataLength))
                        xAccList.append(tempAcc[i][:maxIndex, :])
                        xGyrList.append(tempGyro[i][:maxIndex, :])
                        xMagList.append(tempMag[i][:maxIndex, :])  # Add magnetometer data
                        toAddShape = tempAcc[i][:maxIndex, :].shape[0]
                    elif gyroDataLength >= accDataLength and gyroDataLength >= magDataLength:
                        # If gyroscope is the longest, trim accelerometer and magnetometer data to match its length
                        maxIndex = gyroDataLength - max(abs(gyroDataLength - accDataLength), abs(gyroDataLength - magDataLength))
                        xAccList.append(tempAcc[i][:maxIndex, :])
                        xGyrList.append(tempGyro[i][:maxIndex, :])
                        xMagList.append(tempMag[i][:maxIndex, :])  # Add magnetometer data
                        toAddShape = tempGyro[i][:maxIndex, :].shape[0]
                    else:
                        # If magnetometer is the longest, trim accelerometer and gyroscope data to match its length
                        maxIndex = magDataLength - max(abs(magDataLength - accDataLength), abs(magDataLength - gyroDataLength))
                        xAccList.append(tempAcc[i][:maxIndex, :])
                        xGyrList.append(tempGyro[i][:maxIndex, :])
                        xMagList.append(tempMag[i][:maxIndex, :])  # Add magnetometer data
                        toAddShape = tempMag[i][:maxIndex, :].shape[0]

                    # Add activity labels for each sensor reading
                    yList.append(np.full((toAddShape), activities.index(activity)))


        xAccListClient.append(np.vstack((xAccList)))
        xGyrListClient.append(np.vstack((xGyrList)))
        xMagListClient.append(np.vstack((xMagList)))  # Append magnetometer data
        yListClient.append(np.hstack((yList)))
    clientsAccDataByOrientation.append(np.asarray(xAccListClient, dtype=object))
    clientsGyroDataByOrientation.append(np.asarray(xGyrListClient, dtype=object))
    clientMagDataByOrientation.append(np.asarray(xMagListClient, dtype=object))
    clientsLabelByOrientation.append(np.asarray(yListClient, dtype=object))



# In[ ]:
# conversion to numpy array
clientsAccDataByOrientation = np.asarray(clientsAccDataByOrientation, dtype=object)
clientsGyroDataByOrientation = np.asarray(clientsGyroDataByOrientation, dtype=object)
clientMagDataByOrientation = np.asarray(clientMagDataByOrientation, dtype=object)
clientsLabelByOrientation = np.asarray(clientsLabelByOrientation, dtype=object)

# In[ ]:

# In[ ]:

# stacking all partcipants client
allAcc = np.vstack((np.ravel(clientsAccDataByOrientation)))
allGyro = np.vstack((np.ravel(clientsGyroDataByOrientation)))
allMag = np.vstack((np.ravel(clientMagDataByOrientation)))


# In[ ]:


# Calculating features
meanAcc = np.mean(allAcc)
stdAcc = np.std(allAcc)

meanGyro = np.mean(allGyro)
stdGyro = np.std(allGyro)

meanMag = np.mean(allMag)
stdMag = np.std(allMag)


# In[ ]:


# channel-wise z-normalization
normalizedAcc = (clientsAccDataByOrientation - meanAcc)/stdAcc
normalizedGyro = (clientsGyroDataByOrientation - meanGyro)/stdGyro
normalizedMag = (clientMagDataByOrientation-meanMag)/stdMag


# In[ ]:
stackedOrientationData = []
for normAcc,normGyro,normMag in zip(normalizedAcc,normalizedGyro,normalizedMag):
    stackedOrientationData.append(np.asarray([np.dstack((normAcc,normGyro,normMag)) for normAcc,normGyro,normMag in zip(normAcc,normGyro,normMag)],dtype=object))
stackedOrientationData = np.asarray(stackedOrientationData, dtype=object)


# In[ ]:


dataName = 'RealWorld'
os.makedirs('datasetStandardized_s3/'+dataName, exist_ok=True)
hkl.dump(stackedOrientationData,'datasetStandardized_s3/'+dataName+ '/clientsData.hkl' )
hkl.dump(clientsLabelByOrientation,'datasetStandardized_s3/'+dataName+ '/clientsLabel.hkl' )


# In[ ]:




