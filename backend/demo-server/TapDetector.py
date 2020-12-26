import pandas as pd
import numpy as np

def median_zscore(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[0:lag - 1] = np.ones(lag) * np.median(y[0:lag])
    stdFilter[0:lag - 1] = np.ones(lag) * np.median(np.abs(y[0:lag] - np.median(y[0:lag])))
    for i in range(1, lag):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            signals[i] = 1
            #filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            filteredY[i] = y[i]
        else:
            signals[i] = 0
            filteredY[i] = y[i]
    for i in range(lag, len(y) - 1):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            signals[i] = 1
            #filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            filteredY[i] = y[i]
        else:
            signals[i] = 0
            filteredY[i] = y[i]
        
        avgFilter[i] = (1-influence)*avgFilter[i-1] + influence*np.median(filteredY[(i-lag):i])
        stdFilter[i] = (1-influence)*stdFilter[i-1] + influence*np.median(np.abs(filteredY[(i-lag):i] - np.median(filteredY[(i-lag):i])))
    
    return filteredY, np.asarray(signals)

def detect_taps(acc_df):
  acc_df['magnitude'] = np.linalg.norm(acc_df.loc[:, ['x', 'y', 'z']].values, axis=1)
  acc_df['filtered_magnitude'], acc_df['spike'] = median_zscore(acc_df['magnitude'].values, lag=20, threshold=3.5, influence=0.005)

  taps = []
  skip_to = -1
  for i in acc_df.index:
    if i < skip_to:
      continue
    if acc_df.at[i, 'spike'].any():
      taps.append(i)
      skip_to = i + 400

  taps = np.array(taps)
  return taps

def score_taps(taps, key_df):
  #metrics
  print('Actual taps: ', key_df.shape[0])
  print('Predicted taps: ', taps.shape[0])
  count = 0
  for tap in taps:
    if key_df.loc[(key_df.index >= tap-150) & (key_df.index <= tap+150)].shape[0] >= 1:
      count += 1
  precision = count/taps.shape[0]
  print('Precision: ', precision)
  count = 0
  for tap in key_df.index:
    if ((taps >= tap-150) & (taps <= tap+150)).sum() >= 1:
      count += 1
  recall = count/key_df.shape[0]

  print('Recall: ', recall)
  f1_score = 2 * (precision * recall) / (precision + recall)
  print('F1 score: ', f1_score)

def debug_unseen_taps(unseen_taps, unseen_key_df):
  unseen_taps_true = list()
  for i, tap in enumerate(unseen_taps):
    keys = unseen_key_df.loc[(unseen_key_df.index >= tap-200) & (unseen_key_df.index <= tap+200)]['key'].values
    if len(keys) > 1:
      print(i, 'More than one tap')
    elif len(keys) == 0:
      print(i, 'no tap')
    else:
      if keys[0] != ' ':
        unseen_taps_true.append(i)
      #print(i, keys[0])
  return unseen_taps_true