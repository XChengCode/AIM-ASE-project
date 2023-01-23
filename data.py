
import os
import pandas as pd

def RAVDESS_collate(filepath:str):
    Ravdess = filepath
    ravdess_directory_list = os.listdir(Ravdess)

    file_emotion = []
    file_path = []

    for dir in ravdess_directory_list:
        actor = os.listdir(Ravdess + dir) 
        for file in actor:
            part = file.split('.')[0]
            part = part.split('-')
            file_emotion.append(int(part[2]))
            file_path.append(Ravdess + dir + '/' + file)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    label_df = pd.DataFrame(file_emotion, columns=['Labels'])
    path_df = pd.DataFrame(file_path, columns=['Audio Path'])

    Ravdess_df = pd.concat([emotion_df,label_df,path_df],axis=1)

    Ravdess_df.Emotions.replace(
    {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 
     5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)

    Ravdess_df.Labels.replace(
    {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}, inplace=True)

    return Ravdess_df

def EMODB_collate(filepath:str):
    Emo_DB = filepath
    Emo_DB_directory_list = os.listdir(Emo_DB)
    file_emotion = []
    file_path = []

    for file in Emo_DB_directory_list:
        part = file.split('.')[0]
        part = list(part)
        file_emotion.append(part[5])
        file_path.append(Emo_DB + '/' + file)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    label_df = pd.DataFrame(file_emotion, columns=['Labels'])
    path_df = pd.DataFrame(file_path, columns=['Audio Path'])
    Emo_DB_df = pd.concat([emotion_df,label_df,path_df],axis=1)

    Emo_DB_df.Emotions.replace(
    {'N':'neutral', 'W':'angry', 'A':'fear', 'F':'happy', 'T':'sad', 'E':'disgust', 
    'L':'boredom'}, inplace=True)

    Emo_DB_df.Labels.replace(
    {'N':0, 'W':1, 'A':2, 'F':3, 'T':4, 'E':5, 'L':6}, inplace=True)

    return Emo_DB_df

def CASIA_collate(filepath:str):
    CASIA = filepath
    CASIA_directory_list = os.listdir(CASIA)

    file_emotion = []
    file_path = []

    for actor in CASIA_directory_list:   
        emos = os.listdir(CASIA + '/'+ actor)
        for emo in emos:
            emodir = os.listdir(CASIA + '/'+ actor + '/'+ emo)
            for file in emodir:
                file_emotion.append(emo)
                file_path.append(CASIA + '/'+ actor + '/'+ emo + '/' + file)

    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    label_df = pd.DataFrame(file_emotion, columns=['Labels'])
    path_df = pd.DataFrame(file_path, columns=['Audio Path'])
    CASIA_df = pd.concat([emotion_df,label_df,path_df],axis=1)

    CASIA_df.Labels.replace({'angry':0, 'fear':1, 'happy':2, 
    'neutral':3, 'sad':4, 'surprise':5}, inplace=True)

    return CASIA_df

def Combination_collate(Ravdess_df, Emo_DB_df, CASIA_df):  
    #I did not use this function because of the poor recognition of the combined data set
    
    df_Ravdess_clear = Ravdess_df.drop(Ravdess_df[Ravdess_df['Emotions']=='calm'].index)
    df_Emo_DB_clear = Emo_DB_df.drop(Emo_DB_df[Emo_DB_df['Emotions']=='boredom'].index)
    

    df_Ravdess_clear.Labels.replace(
    {0:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6}, inplace=True)

    df_Emo_DB_clear.Labels.replace(
    {0:0, 1:3, 2:4, 3:1, 4:2, 5:5}, inplace=True)

    CASIA_df.Labels.replace({0:3, 1:4, 2:1, 3:0, 4:2, 5:6}, inplace=True)

    Combination_df = pd.concat([df_Ravdess_clear,df_Emo_DB_clear,CASIA_df],axis=0)
    Combination_df = Combination_df.reset_index()
    Combination_df = Combination_df.drop("index",axis=1)

    return Combination_df