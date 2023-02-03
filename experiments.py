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



def buildID3(df, predicted_column, max_depth=5, path=pd.Series(), current_depth=0, tree=pd.DataFrame()):
    #Calculate the overall entropy of the dataframe
    

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
    
    elif (current_depth / 2) == max_depth:
        path.at[current_depth] = df[predicted_column].mode()[0]
        temp = pd.DataFrame(path).T
        tree = pd.concat([tree, temp], axis=0, ignore_index=True)

    else:
    #Find the next node
        highest_gain = 0
        for column in df.columns.drop([predicted_column]):
            temp_avg_entropy = 0
            for value in df[column].unique():
                temp_df = df[df[column] == value]
                #print(f"{column}: {temp_df[predicted_column].value_counts()[0]}")
                temp_entropy = 0
                for label in temp_df[predicted_column].unique():
                    temp_correct = len(temp_df[temp_df[predicted_column] == label])
                    temp_total = len(temp_df)
                    temp_entropy -= (temp_correct/temp_total) * math.log((temp_correct/temp_total),2)
                #print(f"For {value} in {column}: $$H(Badge) = -({temp_correct}/{total})log_2({temp_correct}/{total}) - ({temp_incorrect}/{total})log_2({temp_incorrect}/{total}) = {round(entropy, 2)}$$")
                temp_avg_entropy += (temp_entropy * temp_total)
            temp_avg_entropy = temp_avg_entropy / len(df)

            info_gain = overall_entropy - temp_avg_entropy
            if info_gain > highest_gain:
                highest_gain = info_gain
                split_on = column
            
        print(f'Splitting on {split_on}. Information gain: {info_gain}')

        path.at[current_depth] = split_on
        current_depth += 1

        for value in np.append(df[split_on].unique(), 'UNKNOWN'):
            if value == 'UNKNOWN':
                path.at[current_depth] = value
                current_depth += 1
                end_this_branch = int(current_depth/2)
                tree = buildID3(df, predicted_column, end_this_branch, path, current_depth, tree)
                path.at[current_depth] = ''
                current_depth -= 1
                path.at[current_depth] = ''
            else:
                path.at[current_depth] = value
                current_depth += 1
                tree = buildID3(df[df[split_on] == value], predicted_column, max_depth, path, current_depth, tree)
                path.at[current_depth] = ''
                current_depth -= 1
                path.at[current_depth] = ''

    return tree


def predict(tree, test_df, predicted_column):
    counter = 0
    test_df['prediction'] = 'Unpredicted'
    #print(len(test_df))
    for i in range (0, len(tree)):
        #temp_df = test_df[test_df[] == tree and test_df[] == tree and test_df[] == tree and test_df[] == tree and test_df[] == tree and test_df[] == tree]
        temp_df = test_df
        for row_num in range(0,int((len(tree.columns)+1)/2)):
            #print(row_num)
            if tree.iloc[i, row_num*2] not in test_df.columns:
                #print(tree.iloc[i, row_num*2])
                for index in temp_df.index:
                    test_df['prediction'].loc[index] = tree.iloc[i, row_num*2]
                break
            else:
                if len(temp_df[temp_df[tree.iloc[i, row_num*2]] == tree.iloc[i, row_num*2+1]]) > 0:
                    temp_df = temp_df[temp_df[tree.iloc[i, row_num*2]] == tree.iloc[i, row_num*2+1]]
                else:
                    temp_df = temp_df[temp_df[tree.iloc[i, row_num*2]] == 'UNKNOWN']
        counter += len(temp_df)
    
    most_common = test_df[predicted_column].mode()[0]
    test_df['prediction'].replace(['Unpredicted'], [most_common], inplace=True)
    #
    max_depth = int((len(tree.columns)-1)/2)
    return test_df, max_depth
    
        
def evaluate_model(df, predicted_column, prediction):
    df['Diff'] = np.where(df[predicted_column] == df[prediction] , 'correct', 'incorrect')
    accuracy = df['Diff'].value_counts()[0]/len(df)
    print(f'Accuracy: {100 * round(accuracy, 4) } %')

    return accuracy











print(f'--Base Line--')
print(f"Accuracy: {100 * round(train_df['label'].value_counts()[train_df['label'].mode()][0]/len(train_df), 4)} %")


tree = buildID3(train_df, 'label', 200)


predictions, max_depth = predict(tree, train_df, 'label')
print(f'Max depth: {max_depth}')
print('on training')
accuracy = evaluate_model(predictions, 'label', 'prediction')

print('on test')
predictions, max_depth = predict(tree, test_df, 'label')

accuracy = evaluate_model(predictions, 'label', 'prediction')


for i in range(1, 6):
    print(f'tested on fold 1: depth {i}')
    tree = buildID3(train1, 'label', i)
    predictions, max_depth = predict(tree, fold1_df, 'label')
    accuracy = evaluate_model(predictions, 'label', 'prediction')

for i in range(1, 6):
    print(f'tested on fold 2: depth {i}')
    tree = buildID3(train2, 'label', i)
    predictions, max_depth = predict(tree, fold2_df, 'label')
    accuracy = evaluate_model(predictions, 'label', 'prediction')

for i in range(1, 6):
    print(f'tested on fold 3: depth {i}')
    tree = buildID3(train3, 'label', i)
    predictions, max_depth = predict(tree, fold3_df, 'label')
    accuracy = evaluate_model(predictions, 'label', 'prediction')

for i in range(1, 6):
    print(f'tested on fold 4: depth {i}')
    tree = buildID3(train4, 'label', i)
    predictions, max_depth = predict(tree, fold4_df, 'label')
    accuracy = evaluate_model(predictions, 'label', 'prediction')

for i in range(1, 6):
    print(f'tested on fold 5: depth {i}')
    tree = buildID3(train5, 'label', i)
    predictions, max_depth = predict(tree, fold5_df, 'label')
    accuracy = evaluate_model(predictions, 'label', 'prediction')
    
