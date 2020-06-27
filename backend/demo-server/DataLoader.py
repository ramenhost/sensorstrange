import pandas as pd

def load_split_session(ses):
  #Sensor data
  sen_df = pd.read_csv('data/' + ses + '/sensor.log')
  init_time = sen_df.at[0,'timestamp']
  sen_df['timestamp'] -= init_time
  sen_df = sen_df.set_index('timestamp')
  #sen_df = sen_df.loc[(sen_df.index > 500) & (sen_df.index < max(sen_df.index)-1000)]

  #Keypress data
  key_df = pd.read_csv('data/' + ses + '/keypress.log')
  key_df['timestamp'] -= init_time
  key_df = key_df.set_index('timestamp')
  key_df = key_df.loc[key_df['action']=='ACTION_DOWN']
  #key_df = key_df.loc[(key_df.index > 1000) & (key_df.index < max(sen_df.index))]
  key_df.dropna(inplace=True)

  periods = key_df.loc[key_df['key'].str.contains('\.')].index
  if(len(periods) < 2):
    raise Exception('Not enough periods in input keypresses')
    
  #train_sen_df = sen_df.loc[sen_df.index<periods[0]]
  unseen_sen_df = sen_df.loc[(sen_df.index>periods[0]) & (sen_df.index<periods[1])]

  #train_acc_df = train_sen_df.loc[train_sen_df['sensor'].str.contains('Acce'), ['x', 'y', 'z']]
  #train_gyro_df = train_sen_df.loc[train_sen_df['sensor'].str.contains('Gyro'), ['x', 'y', 'z']]

  unseen_acc_df = unseen_sen_df.loc[unseen_sen_df['sensor'].str.contains('Acce'), ['x', 'y', 'z']]
  unseen_gyro_df = unseen_sen_df.loc[unseen_sen_df['sensor'].str.contains('Gyro'), ['x', 'y', 'z']]
  
  #Keypress Data
  
  #train_key_df = key_df.loc[key_df.index<periods[0]]
  unseen_key_df = key_df.loc[(key_df.index>periods[0]) & (key_df.index<periods[1])]
  
  return (unseen_acc_df, unseen_gyro_df, unseen_key_df)