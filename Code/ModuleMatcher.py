import cv2
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from ModuleFeatureVector import ColorHist
from ModuleFeatureVector import ComRed
from ModuleFeatureVector import ColorPatches

class Matching:
  debug = False
  url = ''
  
  df = pd.DataFrame()
  keypoints = {}
  descriptors = {}

  flann = cv2.FlannBasedMatcher(dict(algorithm=5, trees=1), dict(checks=100))

  neighComRed = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')
  neighColorPatch = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto')

  lastMatches = []
  roomSequence = []

  connectedRooms = {
    'zaal_A': ['zaal_B','zaal_II'],
    'zaal_B': ['zaal_A','zaal_C','zaal_D'],
    'zaal_C': ['zaal_B','zaal_D'],
    'zaal_D': ['zaal_B','zaal_C','zaal_E','zaal_G','zaal_H'],
    'zaal_E': ['zaal_B','zaal_D','zaal_G','zaal_II'],
    'zaal_F': ['zaal_G','zaal_I','zaal_II'],
    'zaal_G': ['zaal_E','zaal_D','zaal_H','zaal_F','zaal_I'],
    'zaal_H': ['zaal_D','zaal_G','zaal_M'],
    'zaal_I': ['zaal_F','zaal_G','zaal_J','zaal_M','zaal_III'],
    'zaal_J': ['zaal_I','zaal_K'],
    'zaal_K': ['zaal_J','zaal_L'],
    'zaal_L': ['zaal_K','zaal_S','zaal_19','zaal_12','zaal_III'],
    'zaal_M': ['zaal_N','zaal_P','zaal_I','zaal_H'],
    'zaal_N': ['zaal_O','zaal_M'],
    'zaal_O': ['zaal_P','zaal_N'],
    'zaal_P': ['zaal_R','zaal_S','zaal_Q','zaal_M','zaal_O'],
    'zaal_Q': ['zaal_P','zaal_R','zaal_S'],
    'zaal_R': ['zaal_S','zaal_P','zaal_Q'],
    'zaal_S': ['zaal_19','zaal_L','zaal_12','zaal_III','zaal_R','zaal_P','zaal_Q'],
    'zaal_1': ['zaal_2','zaal_II'],
    'zaal_2': ['zaal_1','zaal_3','zaal_4','zaal_5'],
    'zaal_3': ['zaal_2'],
    'zaal_4': ['zaal_2','zaal_5','zaal_7','zaal_8'],
    'zaal_5': ['zaal_2','zaal_4','zaal_7','zaal_II'],
    'zaal_6': ['zaal_7','zaal_9','zaal_II'],
    'zaal_7': ['zaal_4','zaal_5','zaal_6','zaal_8','zaal_9'],
    'zaal_8': ['zaal_4','zaal_7','zaal_13'],
    'zaal_9': ['zaal_6','zaal_7','zaal_10','zaal_III'],
    'zaal_10': ['zaal_9','zaal_11'],
    'zaal_11': ['zaal_10','zaal_12'],
    'zaal_12': ['zaal_11','zaal_19','zaal_L','zaal_S','zaal_III'],
    'zaal_13': ['zaal_8','zaal_14','zaal_16'],
    'zaal_14': ['zaal_13','zaal_15'],
    'zaal_15': ['zaal_14','zaal_16'],
    'zaal_16': ['zaal_13','zaal_15','zaal_17','zaal_18','zaal_19'],
    'zaal_17': ['zaal_16','zaal_18','zaal_19'],
    'zaal_18': ['zaal_16','zaal_17','zaal_19'],
    'zaal_19': ['zaal_12','zaal_16','zaal_17','zaal_18','zaal_S','zaal_L','zaal_III'],
    'zaal_V': ['zaal_II'],
    'zaal_II': ['zaal_V','zaal_1','zaal_5','zaal_6','zaal_A','zaal_E','zaal_F','zaal_III'],
    'zaal_III': ['zaal_I','zaal_9','zaal_S','zaal_L','zaal_12','zaal_19','zaal_II'],
  }


  def __init__(self, keypoints, descriptors, df, url, debug=False):
    if self.debug: print('Init starting...')
    self.debug = debug
    self.url = url + '\\Database'

    # Keypoints
    if self.debug: print('Loading keypoints...')
    self.keypoints = keypoints

    # Descriptors
    if self.debug: print('Loading descriptors...')
    self.descriptors = descriptors

    # DataFrame
    if self.debug: print('Loading DataFrame...')
    self.df = df
    
    # Train ComRed classifier
    if self.debug: print('Training ComRed classifier...')
    self.neighComRed.fit(self.df[['cX', 'cY']], self.df['naam'])

    # Train colorPatch classifier
    if self.debug: print('Training colorPatch classifier...')
    df_temp = pd.DataFrame()
    for i in range(3):
      for j in range(7):
        df_temp[i * 7 + j] = self.df['patch'].apply(lambda array: array[i][j])
    self.neighColorPatch.fit(df_temp, self.df['naam'])
    del df_temp

    if self.debug: print('Init done!')


  def __FlannMatching(self, descr1, descr2):
    matches = self.flann.knnMatch(descr1, descr2, k=2)
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test.
    good = 0
    for i,(m,n) in enumerate(matches):
      if( m.distance < 0.7*n.distance):
        matchesMask[i]=[1,0]
        good += 1

    return good


  def __MatchHist(self, hist_img1, hist_imgn):
    metric_val = cv2.compareHist(hist_img1, hist_imgn, cv2.HISTCMP_CORREL)
    return metric_val


  def MatchPainting(self, img):
    # Weights of classifiers
    weightFlann = 0.65
    weightHist = 0.15
    weightComRed = 0.05
    weightPatches = 0.25

    # Get info from painting
    sift = cv2.SIFT_create()
    key_point, descr = sift.detectAndCompute(img, None)
    hist = ColorHist(img)
    cX, cY = ComRed(img)
    patch = ColorPatches(img)

    # Generate list from where to look
    listOfPaintings = []
    rooms = []
    if len(self.roomSequence) != 0:
      if self.debug: print("Current room: " + self.roomSequence[-1])
      rooms = list(self.connectedRooms.get(self.roomSequence[-1]))
      rooms.append(self.roomSequence[-1])
      for room in rooms:
        paintingsRoom = self.df[self.df.zaal == room].naam
        for painting in paintingsRoom:
          listOfPaintings.append(painting)

    else:
      if self.debug: print("Alle rooms")
      listOfPaintings = self.df['naam']

    df_result = pd.DataFrame(listOfPaintings).rename(columns={0: 'naam'})

    # Flann
    if self.debug: print('flann')
    resultFlann = []
    for index, row in df_result.iterrows():
      resultFlann.append(self.__FlannMatching(self.descriptors[df_result['naam'][index]], descr))
    maxFlann = max(resultFlann)
    df_result['flann'] = [float(i/maxFlann) for i in resultFlann]
    del resultFlann

    # Histogram
    if self.debug: print('histo')
    resultHisto = []
    for index, row in df_result.iterrows():
      resultHisto.append(self.__MatchHist(self.df.loc[self.df['naam'] == df_result['naam'][index]]['hist'].iloc[0], hist))
    df_result['hist'] = [abs(i) for i in resultHisto]
    del resultHisto

    # Classifier ComRed
    if self.debug: print('comred')
    result = self.neighComRed.predict_proba([[cX, cY]])[0]
    dic = {self.neighComRed.classes_[i]: result[i] for i in range(len(self.neighComRed.classes_))}
    df_temp = pd.DataFrame(list(dic.items())).rename(columns={0: 'naam', 1: "comred"})
    df_result = pd.merge(df_result, df_temp, on='naam')
    del dic, df_temp

    # Classifier Patches
    if self.debug: print('patches')
    temp = []
    for i in range(3):
      for j in range(7):
        temp.append(patch[i][j])
    result = self.neighColorPatch.predict_proba([temp])[0]
    del temp
    dic = {self.neighColorPatch.classes_[i]: result[i] for i in range(len(self.neighColorPatch.classes_))}
    df_temp = pd.DataFrame(list(dic.items())).rename(columns={0: 'naam', 1: "patches"})
    df_result = pd.merge(df_result, df_temp, on='naam')
    del dic, df_temp

    # Make combined classification
    df_result['total'] = df_result['flann'] * weightFlann + df_result['hist'] * weightHist + df_result['comred'] * weightComRed + df_result['patches'] * weightPatches
    print(df_result.sort_values(by=['total'], ascending=False)[:5]) #REMOVE
    df_result = df_result.sort_values(by=['total'], ascending=False)[:1]

    # Append to list
    room = df_result['naam'].values[0].split('__')[0]
    room = room[0].lower() + room[1:]

    if len(self.lastMatches) == 5:
      self.lastMatches.pop(0)
    self.lastMatches.append(room)

    if len(self.roomSequence) == 0: 
      self.roomSequence.append(room)
    else: 
      if self.roomSequence[-1] != room: 
        self.roomSequence.append(room)
        
    return df_result