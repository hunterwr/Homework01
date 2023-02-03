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

print(f'--Base Line--')
print(f"Accuracy: {round(train_df['label'].value_counts()[train_df['label'].mode()][0]/len(train_df), 4)}")

