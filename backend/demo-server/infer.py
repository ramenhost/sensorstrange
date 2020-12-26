from DataLoader import load_split_session
from TapDetector import detect_taps, debug_unseen_taps, score_taps
from RegionClassifier import train_hypo, classify_taps, score_classification
from LanguageModel import LanguageModel

import os

def pipeline(train_ses, unseen_ses, lm):
  #print(train_ses, unseen_ses)
  train_acc_df, train_gyro_df, train_key_df = load_split_session(train_ses)
  unseen_acc_df, unseen_gyro_df, unseen_key_df = load_split_session(unseen_ses)
  
  print('Detecting Taps')
  train_taps = detect_taps(train_acc_df)
  unseen_taps = detect_taps(unseen_acc_df)
  unseen_taps_true = debug_unseen_taps(unseen_taps, unseen_key_df)
  # print("Train Data:")
  # score_taps(train_taps, train_key_df)
  # print("Unseen Data:")
  score_taps(unseen_taps, unseen_key_df)
  
  print('Classifing keyboard regions')
  hypo, key_count = train_hypo(train_acc_df, train_gyro_df, train_key_df, train_taps)
  y_pred, y_pred_old = classify_taps(hypo, unseen_taps, unseen_acc_df, unseen_gyro_df, key_count)
  # print("With scoring")
  score_classification(unseen_taps, unseen_key_df, y_pred)
  print(y_pred)
  # print("Without scoring")
  # score_classification(unseen_taps, unseen_key_df, y_pred_old)
  # print(y_pred_old)
  region_seq = [y_pred[i] for i in unseen_taps_true]
  
  print('Generating probable words')
  inferred = lm.get_probable_words(region_seq)

  return inferred

def loop_files(train_file, test_files):
  lm = LanguageModel('corpus/google-10000-english.txt')
  lm.load_weights('model/keras_char_rnn.500.h5')
  found = 0
  for test_file in test_files:
    print('-------------------------')
    print('Starting for ', test_file)
    result = pipeline(train_file, test_file, lm)
    print(result)
    actual = test_file.split('_')[0]
    if actual in result:
      print('Found', actual)
      found += 1

  print('Inference rate: ', found/len(test_files))

if __name__=='__main__':
  print('-------------------------------------')
  print('Text Inference from Smartphone motion')
  print('-------------------------------------')
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

  # loop_files('asw_v2', ['music_1', 'music_2', 'music_3', 'music_4', 'music_5', 'music_6', 'music_7', 'music_8', 'music_9', 'music_10', 'flight_1', 'flight_2', 'black_1', 'black_2', 'glass_1', 'glass_2', 'movie_1', 'movie_2', 'program_1', 'program_2', 'sport_1', 'sport_2', 'ticket_1', 'ticket_2'])
  # exit(0)

  lm = LanguageModel('corpus/google-10000-english.txt')
  lm.load_weights('model/keras_char_rnn.500.h5')
  
  print('Enter train and test files seperated by comma')

  while(True):
    data_file = input('>').split(',')
    if data_file[0] == 'exit' or data_file[0] == 'quit':
      break
    train_file, test_file = data_file[0], data_file[1]
    result = pipeline(train_file, test_file, lm)
    print(result)