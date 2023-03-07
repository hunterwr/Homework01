import pandas as pd
import numpy as np
import math


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
fold1_df = pd.read_csv('fold1.csv')
fold2_df = pd.read_csv('fold2.csv')
fold3_df = pd.read_csv('fold3.csv')
fold4_df = pd.read_csv('fold4.csv')
fold5_df = pd.read_csv('fold5.csv')


train1 = pd.concat([fold5_df, fold2_df, fold3_df, fold4_df])
train2 = pd.concat([fold5_df, fold1_df, fold3_df, fold4_df])
train3 = pd.concat([fold5_df, fold2_df, fold1_df, fold4_df])
train4 = pd.concat([fold5_df, fold2_df, fold3_df, fold1_df])
train5 = pd.concat([fold1_df, fold2_df, fold3_df, fold4_df])




def calc_entropy(df, predicted_column):
    overall_entropy = 0
    for label in df[predicted_column].unique():
        correct = len(df[df[predicted_column] == label])
        overall_total = len(df)
        overall_entropy -= (correct/overall_total) * math.log((correct/overall_total),2)
    if overall_entropy == 0:
        path.at[current_depth] = label
        #print(path)
        temp = pd.DataFrame(path).T
        #display(temp)
        tree = pd.concat([tree, temp], axis=0, ignore_index=True)