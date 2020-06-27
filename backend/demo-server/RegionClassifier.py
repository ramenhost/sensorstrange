import numpy as np

def interpolate(df, start, end, sample_interval=1):
  X2 = [i for i in range(start, end+1, sample_interval)]
  interval = df.loc[(df.index >= start) & (df.index <= end)]
  X1 = interval.index
  xY1 = interval.loc[:,'x'].values
  yY1 = interval.loc[:,'y'].values
  zY1 = interval.loc[:,'z'].values
  xY2 = np.interp(X2, X1, xY1).reshape(-1, 1)
  yY2 = np.interp(X2, X1, yY1).reshape(-1, 1)
  zY2 = np.interp(X2, X1, zY1).reshape(-1, 1)
  return np.hstack((xY2, yY2, zY2))

region_split = {
  1: ['Q', 'W', 'A', 'S'],
  2: ['E', 'R', 'D', 'F'],
  3: ['Z', 'X'],
  4: ['T', 'Y', 'U', 'G', 'H', 'J'],
  5: ['I', 'O', 'P', 'K', 'L', 'M'],
  6: ['C', 'V', 'B', 'N', ' ']
}

region_split_overlap = {
  1: ['Q', 'W', 'A', 'S', 'E', 'Z'],
  2: ['E', 'R', 'D', 'F', 'S', 'T'],
  3: ['Z', 'X', 'S', 'D', 'C'],
  4: ['T', 'Y', 'U', 'G', 'H', 'J', 'R', 'I', 'K'],
  5: ['I', 'O', 'P', 'K', 'L', 'M', 'J', 'B'],
  6: ['C', 'V', 'B', 'N', ' ', 'X', 'M']
}

def key_area(key):
  for region, keys in region_split.items():
    if key in keys:
      return region
  raise Exception('Unknown key' + str(key))

standardize = lambda x: (x-np.mean(x, axis=0))/np.std(x, axis=0)

def train_hypo(acc_df, gyro_df, key_df, taps):
  standardize = lambda x: (x-np.mean(x, axis=0))/np.std(x, axis=0)

  hypo = dict()
  key_count = dict()

  for tap in taps:

    #Cut window from 75ms before tap to 125ms after tap
    interval_start, interval_end = tap-200, tap+200
    try:
      data = standardize(np.hstack((interpolate(acc_df, interval_start, interval_end), interpolate(gyro_df, interval_start, interval_end))))
    except:
      continue
    
    #Label key region
    keys = key_df.loc[(key_df.index >= interval_start) & (key_df.index <= interval_end)]['key'].values
    if len(keys) == 0:
      continue
    try:
      key = key_area(keys[0])
    except:
      continue

    #Store 6D vectors in a dictionary
    if key in hypo:
      hypo[key].append(data)
      key_count[key]+=1
    else:
      hypo[key]=list()
      hypo[key].append(data)
      key_count[key]=1
      
  return hypo, key_count

def classify_taps(hypo, taps, acc_df, gyro_df, key_count):
  
  #l2_norm = lambda x, y: (x - y) ** 2
  rms = lambda x, y: np.sqrt(np.mean((x-y)**2))

  #Sum of individial axis DTW (performs better than combined)
  def doRMS(data1, data2):
    d = 0
    for i in range(6):
      d_temp = rms(data1[:, i].reshape(-1, 1), data2[:, i].reshape(-1, 1))
      d += d_temp
    return d
  
  
  y_pred = list()
  y_pred_old = list()
  l_count = min(key_count.values())

  for tap in taps:
    tap_start = tap - 200
    tap_end = tap + 200
    data = standardize(np.hstack((interpolate(acc_df, tap_start, tap_end), interpolate(gyro_df, tap_start, tap_end))))
    distances = list()
    for label, l_hypos in hypo.items():
      l_distances = list()
      for l_hypo in l_hypos:
        l_distances.append([doRMS(data, l_hypo), label])
      l_distances = np.array(l_distances)
      l_distances = l_distances[l_distances[:, 0].argsort()]
      l_distances = l_distances[:5]
      distances.extend(l_distances)

    dis = np.array(distances)
    dis = dis[dis[:,0].argsort()]
    dis = dis[:7]

    weight = dis.shape[0]
    scores = [0] * len(region_split)
    for reg in dis[:, 1]:
      scores[int(reg)-1] += weight
      weight -= 1

    pred = np.array(scores).argmax() + 1

    y_pred.append(pred)
    y_pred_old.append(int(dis[0, 1]))
    
  return y_pred, y_pred_old

def score_classification(taps, key_df, y_pred):

  def is_key_in_reg(key, reg):
    return (key in region_split_overlap[reg])

  score = list()
  region_seq = y_pred[1:-1]
  for i, tap in enumerate(taps[1:-1]):
    keys = key_df.loc[(key_df.index >= tap-200) & (key_df.index <= tap+200)]['key'].values
    if len(keys) == 0:
      score.append(0)
    else:
      if is_key_in_reg(keys[0], region_seq[i]):
        score.append(1)
      else:
        score.append(0)
  print('accuracy: ', sum(score)/len(score))
#  print('accuracy: ', accuracy_score(y_test, y_pred))
  #Confusion matrix
#   print('Confusion matrix')
#   labels = [1, 2, 3, 4, 5]
#   print(confusion_matrix(y_test, y_pred, labels))