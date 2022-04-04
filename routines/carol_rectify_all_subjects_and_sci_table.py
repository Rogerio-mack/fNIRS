# -*- coding: utf-8 -*-
"""Carol_rectify_all_subjects_and_sci_table.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/Rogerio-mack/fNIRS/blob/main/Carol_rectify_all_subjects_and_sci_table.ipynb

## Mount Google Drive
"""

#@markdown
from google.colab import drive
drive.mount('/content/drive')

"""## Common Imports"""

#@markdown 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

import h5py
from itertools import compress
from sklearn.preprocessing import scale
from google.colab import widgets

"""## Install & import `nilearn` and `mne-nirs`"""

#@markdown
!pip install nilearn
!pip install -U --no-deps https://github.com/mne-tools/mne-python/archive/main.zip
!pip install mne-nirs

from mne.io import read_raw_nirx
from mne.preprocessing.nirs import (optical_density, beer_lambert_law,
                                    temporal_derivative_distribution_repair)
import mne

"""## Routines"""

#@markdown `def clone_snirf(path_in, path_out, fname='out.snirf')`
def clone_snirf(path_in, path_out, fname='out.snirf'):
  
  if path_in == path_out:
    print('Error: path_in and path_out can not be the same.') 
    return

  if not os.path.exists(path_out):
      os.mkdir(path_out)
      print("Directory " , path_out ,  "was created")
  else:    
      print("Warning: directory " , path_out ,  " already exists")

  if os.path.exists(path_out + fname):
    os.remove(path_out + fname)
    print('Warning: previous output file was deleted.')
  
  print('Input snirf file: ' + path_in + fname)

  fs = h5py.File(path_in + fname,'r')
  fd = h5py.File(path_out + fname,'w')

  if list(fs.keys()).count('formatVersion') >= 1:
    fs.copy('formatVersion',fd,'formatVersion') 
  else:
    print('Warning: there is no formatVersion key in snirf input.')

  if list(fs.keys()).count('nirs') >= 1:
    fs.copy('nirs',fd,'nirs') 
  else:
    print('Error: Invalid snirf file. There is no nirs key in snirf input.')  
  
  print('Output snirf file: ' + path_out + fname)
  
  fd.close()
  fs.close()

  return

#@markdown `def create_channels_table(data,reorder=False)`
def create_channels_table(data,reorder=False):

  if reorder:
    print('Warning: reorder=True puts channels names in ascendent order, but this can be a different order from snirf file.')

  # Parameters: 
  # data: snirf file read

  # create alias for all data['nirs'].keys()
  for key in data['nirs'].keys():
    exp = key + ' = data["nirs"]["' + key + '"] '
    print(exp)
    exec(exp)
    
  measurements = [ x for x in list( data1.keys() ) if x.find('measurementList') == 0 ]

  measure_idx = []
  name = []
  source =  []
  detector = []
  source_pos = []
  detector_pos = []
  source_pos_3D = []
  detector_pos_3D = []
  flag_2D = True
  flag_3D = True
  frequencies = []

  for m in measurements:
    measure_idx.append( int( m[ len('measurementList'):: ] ) )
    
    e_name = probe['sourceLabels'][ int(data1[m]['sourceIndex'][0]) - 1 ].decode('UTF-8') + '_' + probe['detectorLabels'][ int(data1[m]['detectorIndex'][0]) - 1 ].decode('UTF-8') 
    e_source = int(data1[m]['sourceIndex'][0])   
    e_detector = int(data1[m]['detectorIndex'][0]) 
    
    if list(probe.keys()).count('sourcePos2D') > 0 and list(probe.keys()).count('detectorPos2D') > 0: 
      e_source_pos = probe['sourcePos2D'][ int(data1[m]['sourceIndex'][0]) - 1 ] # .decode('UTF-8')
      e_detector_pos = probe['detectorPos2D'][ int(data1[m]['detectorIndex'][0]) - 1 ] # .decode('UTF-8')
    else:
      flag_2D = False
      e_source_pos = e_detector_pos = np.array([0,0],dtype=float)

    # e_frequencies = probe['frequencies'][0]

    if list(probe.keys()).count('sourcePos3D') > 0 and list(probe.keys()).count('detectorPos3D') > 0:
      e_source_pos_3D = probe['sourcePos3D'][ int(data1[m]['sourceIndex'][0]) - 1 ] # .decode('UTF-8')
      e_detector_pos_3D = probe['detectorPos3D'][ int(data1[m]['detectorIndex'][0]) - 1 ] # .decode('UTF-8')
    else:
      flag_3D = False
      e_source_pos_3D = e_detector_pos_3D = np.array([0,0,0],dtype=float)

    name.append( e_name )
    source.append( e_source )
    detector.append( e_detector )
    source_pos.append( e_source_pos )
    detector_pos.append( e_detector_pos )
  #  frequencies.append( e_frequencies )
    source_pos_3D.append( e_source_pos_3D )
    detector_pos_3D.append( e_detector_pos_3D )

  if not flag_2D:
    print('WARNING: no source, detector pos 2D available\n')

  if not flag_3D:
    print('WARNING: no source, detector pos 3D available\n')

  # print(measure_idx)  
  # print(name)  
  # print(source)
  # print(detector)
  # print(source_pos)
  # print(detector_pos)
  # print(frequencies)
  # print(source_pos_3D)
  # print(detector_pos_3D)

  measures = pd.DataFrame({'measure_idx':measure_idx,
                          'name':name,
                          'source':source,
                          'detector':detector,
  #                         'frequencies':frequencies,
                          'source_pos':source_pos,
                          'detector_pos':detector_pos,
                          'source_pos_3D':source_pos_3D,
                          'detector_pos_3d':detector_pos_3D}) # .sort_values('measure_idx').reset_index(drop=True)
  # measures.head()

  wavelength = []

  for m in measurements:
    wavelength.append( str(int(probe['wavelengths'][ int(data1[m]['wavelengthIndex'][0]) - 1 ])) )
  #  print( m,str(int(probe['wavelengths'][ int(data1[m]['wavelengthIndex'][0]) - 1 ]))  )

  # wavelength
  measures['wavelength'] = wavelength 
  measures['name'] = measures['name'] + '_' + wavelength 
  measures.set_index('measure_idx',inplace=True) 

  measures = measures[['name', 'source', 'detector', 'wavelength', 'source_pos', 'detector_pos',
        'source_pos_3D', 'detector_pos_3d']]
  measures = measures.rename_axis('')
  measures.head()

  if reorder:
    measures = measures.sort_values('name')

  return measures.reset_index(drop=True)

#@markdown `def create_channels_raw(data,reorder=False)`
def create_channels_raw(data,reorder=False):

  # Parameters: 
  # data: snirf file read

  if reorder:
    print('Warning: reorder=True puts channels names in ascendent order, but this can be a different order from snirf file.')

  # create alias for all data['nirs'].keys()
  for key in data['nirs'].keys():
    exp = key + ' = data["nirs"]["' + key + '"] '
    exec(exp)

  measurements = [ x for x in list( data1.keys() ) if x.find('measurementList') == 0 ]

  measure_idx = []
  name = [] 
  for m in measurements:
    measure_idx.append( int( m[ len('measurementList'):: ] ) )
    e_name = probe['sourceLabels'][ int(data1[m]['sourceIndex'][0]) - 1 ].decode('UTF-8') + '_' + probe['detectorLabels'][ int(data1[m]['detectorIndex'][0]) - 1 ].decode('UTF-8') 
    name.append( e_name )

  measures = pd.DataFrame({'name':name})

  wavelength = []

  for m in measurements:
    wavelength.append( str(int(probe['wavelengths'][ int(data1[m]['wavelengthIndex'][0]) - 1 ])) )
  #  print( m,str(int(probe['wavelengths'][ int(data1[m]['wavelengthIndex'][0]) - 1 ]))  )

  measures['name'] = measures['name'] + '_' + wavelength 

  # print(measures)

  raw = pd.concat([pd.DataFrame(data1['time']), pd.DataFrame(data1['dataTimeSeries'])], axis=1)
  raw.columns = ['time'] + list(measures.name) 
  
  if reorder:
    df_temp = raw.drop(columns='time')
    df_temp = df_temp.reindex(sorted(df_temp.columns), axis=1)
    df_temp = pd.concat([raw.time,df_temp], axis=1 )
    raw = df_temp 

  return raw

#@markdown `def normalize_channels(df)`
def normalize_channels(df):
  df_scaled = df.drop(columns='time').copy()
  
  for c in df_scaled:
    df_scaled[c] = scale( df_scaled[c] )

  df_scaled = pd.concat([df.time,df_scaled],axis=1)
  return df_scaled 

# normalize_channels(channels_raw)

#@markdown `def plotchannels_all(df, reds=True, normalize=False, figsize=(20,16), cv_threshold=0.25)`
def plotchannels_all(df, reds=True, normalize=False, figsize=(20,16), cv_threshold=0.25):

  f = plt.figure( figsize=figsize )

  if normalize:
    df_not_scaled = df.copy()
    df = normalize_channels(df)
  else:
    df_not_scaled = df # the same

  inicio = 0  
  inc = 6
  add_inc = 0
  ticks = []

  col_order = df.drop(columns='time').columns.sort_values()

  for c in col_order:
    if not reds:
      sns.lineplot( x=df['time'], y=df[c] + inicio + add_inc )
    else:
      if df_not_scaled[c].min() < 0:
        sns.lineplot( x=df['time'], y=df[c] + inicio + add_inc, color='r', alpha=0.8)
      else:
        if df_not_scaled[c].mean() != 0:
          cv = df_not_scaled[c].std() / df_not_scaled[c].mean()
        else:
          cv = 0
        sns.lineplot( x=df['time'], y=df[c] + inicio + add_inc, color=['gray','yellow'][cv > cv_threshold or cv == 0], alpha=[0.5,0.8][cv > cv_threshold or cv == 0])

    ticks.append(inicio + add_inc)
    add_inc = add_inc + inc
    
  ylim = f.gca().get_ylim()[1]

  # for c in group_nirs['group']['stimulus'].drop(columns='time'):
  #  plt.plot(stimulus['time'],stimulus[c]*ylim,label=c,lw=1,linestyle='dashed')

  # plt.legend()
  if normalize:
    plt.yticks(ticks = ticks, labels = list( col_order ))
    plt.ylabel('')
  else:
    plt.ylabel('Channel Values')

  plt.show()

  return 

# plotchannels_all(channels_raw)
# plotchannels_all(channels_raw,normalize=True,figsize=(10,10))

#@markdown `def check_sample_rate(df)`
def check_sample_rate(df):

  sample_rate = np.float(df.time[-1:] / len(df.time))
  print('Sample rate: ', sample_rate, 'in sec.')

  total_time =  df.time[-1:].values[0]
  print('Total time: ',   total_time , 'in sec.')

  resample = None 

  if not all( channels_raw.time.diff()[1::]*10 == channels_raw.time.diff()[1]*10 ):
    print('Warning: time seems have different intervals. For better signal processing sample rate should be constant. Try to check and correct this.')
    print('Sample rate suggested: 0.1 sec (for this, you can use time values of the third return of this function)')

    resample_time = np.round( np.arange(0,len(df),1)/10 , 1)

  return sample_rate, total_time, resample_time

# sample_rate, total_time, resample_time = check_sample_rate(channels_raw)

# to change...
# channels_raw.time = resample_time
# sample_rate, total_time, resample_time = check_sample_rate(channels_raw)

#@markdown `def perc_negatives(s_neg)`
def perc_negatives(s_neg):
#
# Parameters:
# s_neg: np.array with negatives values
#
# Returns:
# perc_neg: % of negatives values
#
# Test:
# perc_negatives(np.array([1,2,-1,5,5,7,-1,-1,10],dtype='float'))
#
  return len( np.array(np.where( s_neg < 0 )).ravel() ) / len(s_neg)

# perc_negatives(np.array([1,2,-1,5,5,7,-1,-1,10],dtype='float'))

# channels_raw.drop(columns='time').apply(perc_negatives)

#@markdown `def plotchannels_tab(df, reds=False, threshold_neg = 0.10, all=True, normalized=True, statistics=True, individuals=True,  cv_threshold=0.25)`
# from matplotlib import pylab
from google.colab import widgets

def plotchannels_tab(df, reds=False, threshold_neg = 0.10, all=True, normalized=True, statistics=True, individuals=True,  cv_threshold=0.25):

  chnames = df.drop(columns='time').columns.sort_values()

  if individuals:
    tabnames = chnames.to_list()

    # Mark tabnames with '()' for channels with negative values
    for i in range(len(tabnames)):
      if df[ tabnames[i] ].min() < 0:
        tabnames[i] = '(' + tabnames[i] + ')'
  else:
    tabnames = []

  # Add other tabs
  before = 0
  
  if normalized:
    tabnames = ['Normalized'] + tabnames 
    before = before + 1

  if all:
    tabnames = ['All'] + tabnames 
    before = before + 1

  if statistics:
    tabnames = tabnames + ['Statistics']

  tb = widgets.TabBar(tabnames)

  if all:
    with tb.output_to(tabnames.index('All')):
      plotchannels_all(df,reds=True)

  if normalized:
    with tb.output_to(tabnames.index('Normalized')):
      plotchannels_all(df,reds=True,normalize=True)

  if individuals:
    for i in range(len(chnames)):
      # Only select the first 2 tabs, and render others in the background.
      with tb.output_to(i+before, select=(i+before < before)): # +2 because ['All','Normalized']
        if reds:
          if df[chnames[i]].min() < 0:          
            plt.plot(df.time, df[chnames[i]], color=['blue','red'][df[chnames[i]].min() < 0] )
          else:
            if df[chnames[i]].mean() != 0:
              cv = df[chnames[i]].std() / df[chnames[i]].mean()
            else:
              cv = 0  
            plt.plot(df.time, df[chnames[i]], color=['blue','yellow'][cv > cv_threshold or cv == 0] )
        else:
          plt.plot(df.time, df[chnames[i]], color='blue')

  if statistics:
    with tb.output_to(tabnames.index('Statistics'),select=0):
      display( df.drop(columns='time').describe() )
      print()

      neg_true = df.drop(columns='time').apply(min) < 0  

      if any(neg_true):
        print('Warning: There are negative values in signals.')

      for c in neg_true[ neg_true == True ].index:
        if perc_negatives(df[c]) > threshold_neg:
          print(c,'\t...had negative values over ', threshold_neg, ' rate, values should be rescaled to positive')
        else:
          print(c,'\t...had negative values, values should be interpolated')

      print('Warning: Just check sample rate here.')
      check_sample_rate(df)

      print()
      print('Coeficientes de Variação, Threshold = ', cv_threshold, ' e Presença de Valores Negativos')
      
      cv_list = [] # 0 for channels with all values 0 (invalid)
      for c in channels_raw.drop(columns = 'time'):
        if channels_raw.drop(columns = 'time')[c].mean() != 0:
          cv_list.append( channels_raw.drop(columns = 'time')[c].std() / channels_raw.drop(columns = 'time')[c].mean() )
        else:
          cv_list.append(0)  
      df_cv = pd.DataFrame( channels_raw.drop(columns = 'time').std() ) # just to inicialize df
      df_cv.rename(columns={0:'coef_var'},inplace=True)
      df_cv['coef_var'] = cv_list # replace with real cvs
      df_cv['up_threshold'] = df_cv['coef_var'] > cv_threshold
      df_cv['negative_values'] = channels_raw.drop(columns = 'time').min() < 0
      print(df_cv)

      print()
      print('*Coeficientes de Variação é válido somente para valores positivos')

  return

#@markdown `def interpolate_negatives(s_neg, threshold_neg = 0.10)`
def interpolate_negatives(s_neg, threshold_neg = 0.10):
#
# Parameters:
# s_neg: np.array with less than threshold_neg of negatives values to be interpolate
# threshold_neg: 0.10
#
# Returns:
# s_pos: np.array with values interpolated
#
# Test:
# interpolate_negatives(np.array([1,2,-1,5,5,7,-1,-1,10],dtype='float'))
#
  if perc_negatives(s_neg) > threshold_neg:
    print(c,'Error: there are more negative values than threshold_neg. You can try rerun with a different threshold.')
    return None

  xp = np.array(np.where( s_neg >= 0 )).ravel()
  fp = s_neg[ s_neg >= 0 ]
  x = np.array(np.where( s_neg < 0 )).ravel()
  s_new = np.interp(x, xp, fp)

  s_pos = s_neg.copy()
  s_pos[ s_pos < 0 ] = s_new

  return s_pos

# interpolate_negatives(np.array([1,2,-1,5,5,7,-1,-1,10],dtype='float'))

#@markdown `def rectify_negatives(df, threshold_neg = 0.10, transform='zeros')`
def rectify_negatives(df, threshold_neg = 0.10, transform='zeros'):
 
  neg_true = df.drop(columns='time').apply(min) < 0  

  rectified_negatives = neg_true[ neg_true == True ].index

  for c in rectified_negatives:
    if perc_negatives(df[c]) > threshold_neg:
      if transform == 'zeros':
        df[c] = 0
        print(c,'\t...had values transformed to zeros')
      elif transform == 'positive':
        df[c] = df[c] + np.abs(df[c].min())
        print(c,'\t...had values rescaled to positive')  
      else:
        print('parameter transform erro. Only transforms negatives values to "zeros" or "positive" available up to now')  
    else:
      df[c] = interpolate_negatives(np.array(df[c]))
      print(c,'\t...had negative values interpolated')

  return df, list(rectified_negatives)

#@markdown `def add_appname_snirf(data, appname='snirf-Mack')`
def add_appname_snirf(data, appname='snirf-Mack'):

  data1 = data['nirs']['data1'] 

  if list(metaDataTags.keys()).count('AppName') == 1:
    metaDataTags['AppName'][...] = [appname]
  else:
    metaDataTags['AppName'] = np.array([appname],dtype='|S13')

  return

#@markdown `def rectify_negatives_snirf(data, channels_raw, rectified_negatives)` 
def rectify_negatives_snirf(data, channels_raw, rectified_negatives):

  data1 = data['nirs']['data1'] 

  for c in rectified_negatives:
  # print( c, list(channels_raw.columns).index(c) - 1 )
  # print( data1['dataTimeSeries'][:, list(channels_raw.columns).index(c) - 1 ] )
  # print( any(data1['dataTimeSeries'][:, list(channels_raw.columns).index(c) - 1 ] < 0) )   
    data1['dataTimeSeries'][:, list(channels_raw.columns).index(c) - 1 ] = channels_raw[c]  

  return

#@markdown `def rectify_time_snirf(data, channels_raw)`
def rectify_time_snirf(data, channels_raw):

  data1 = data['nirs']['data1'] 

  data1['time'][...] = channels_raw.time
  # print( data1['time'][...] )

  return

#@markdown `def create_stim_from_aux(data,delete_before=False,show=None)`
def create_stim_from_aux(data,delete_before=False,show=None):

  stim_list = [x for x in list(data['nirs'].keys()) if x.find('stim') == 0]

  if (len(stim_list)) > 0 and delete_before:
    for s in stim_list:
      del data['nirs'][s]
      print('Warning: previous stim record ' + s + ' was deleted.')

  if len(stim_list) > 0 and not delete_before:
    print('Error: previous stim records ', stim_list,  ' should be deleted before create stim from aux.')
    return

  aux_list = [x for x in list(data['nirs'].keys()) if x.find('aux') == 0]
  if len(aux_list) == 0:
    print('Error: there is no aux records in data.')
    return 
    
#  for a in aux_list:
#    print( data['nirs'][a].keys() )

  stim_aux = False

  for a in aux_list:
    if list(data['nirs'][a].keys()) == ['dataTimeSeries', 'name', 'time', 'timeOffset']: 
      stim_aux = True

  if not stim_aux:
    print('Error: it seems there is no valid aux stim records.')
    return
  else:
    print('Warning: trying to use aux stim records for stimulus. At least one record with dataTimeSeries, name, time and timeOffset.') 

  for a in aux_list:
    data['nirs'][a]['time'][...] = data1['time'][:] # ajusta o tempo
    data['nirs'][a]['dataTimeSeries'][:] = np.round(   data['nirs'][a]['dataTimeSeries'][:], 0) # ajusta o impulso

  for a in aux_list:
    stim = 'stim' + a.split('aux')[1]
    data['nirs'].create_group(stim)
    data['nirs'][stim].create_dataset('name', data=np.array([data['nirs'][a]['name'][0]]).astype('|O'))

    start = []
    end = []
    value = []

    start_bool = False

    for i in range( len( aux1['dataTimeSeries'] ) ):
      if data['nirs'][a]['dataTimeSeries'][i] != 0 and not start_bool:
        value.append( data['nirs'][a]['dataTimeSeries'][i] )
        start.append( data['nirs'][a]['time'][i] )
        start_bool = True
      else: 
        if data['nirs'][a]['dataTimeSeries'][i] == 0 and start_bool:  
          end.append( data['nirs'][a]['time'][i-1] )
          start_bool = False

    data['nirs'][stim].create_dataset('data', data=np.array( [np.array(start) , np.array(end) - np.array(start) , np.array(value)] ).T)

  for a in aux_list:
    if data['nirs'][a]['timeOffset'][0] != 0:
      print('Warning: aux record ' + a + ' with timeOffset different from 0. timeOffset will be ignored when transformed to stim records.')

  stim_list = [x for x in list(data['nirs'].keys()) if x.find('stim') == 0]
  
  print('Stim records was created: ', stim_list)

  if show != None:
    print('Showing stim record ', show, '...')
    print(data['nirs'][show]['name'][0])
    print(data['nirs'][show]['data'][:])

  return

#@markdown `def plot_aux_stimulus(data)`
def plot_aux_stimulus(data):

  aux_list = [x for x in list(data['nirs'].keys()) if x.find('aux') == 0]
  if len(aux_list) == 0:
    print('Error: there is no aux records in data.')
    return 

  f = plt.figure(figsize=(20,4))

  for a in aux_list:
    plt.plot(data['nirs']['data1']['time'][:], data['nirs'][a]['dataTimeSeries'][:],label=a)

  plt.legend()
  plt.show()

  return

#@markdown `def rectify_stim_duration(data,duration=5,show=None)`
def rectify_stim_duration(data,duration=5,show=None):

  stim_list = [x for x in list(data['nirs'].keys()) if x.find('stim') == 0]

  if (len(stim_list)) == 0:
    print('Error: there is no stim records in data.')
    return

  for s in stim_list:
    data['nirs'][s]['data'][:,1] = np.ones( data['nirs'][s]['data'].shape[0] )*duration
  
  if show != None:
    print('Showing stim record ', show, '...')
    print(data['nirs'][show]['name'][0])
    print(data['nirs'][show]['data'][:])

  return

#@markdown `def create_stim_rest(data,show=True)`
def create_stim_rest(data,show=True):

  stim_list = [x for x in list(data['nirs'].keys()) if x.find('stim') == 0]

  if (len(stim_list)) == 0:
    print('Error: there is no stim records in data.')
    return

  stim = 'stim' + str( len(stim_list)+1 )
  data['nirs'].create_group(stim)
  data['nirs'][stim].create_dataset('name', data=np.array(['rest']).astype('|O'))

  start = []
  start_stim = []
  end = []
  value = []

  start_bool = False

  for s in stim_list:
    for i in range( data['nirs'][s]['data'].shape[0] ):
      start_stim.append( data['nirs'][s]['data'][i,0] )
      start.append( data['nirs'][s]['data'][i,0] + data['nirs'][s]['data'][i,1] )
      value.append(-1)

  start_stim = sorted(start_stim)
  start = sorted(start)

  for i in range(len(start_stim) - 1):
    end.append( start_stim[i+1] )

  start = start[0:-1]
  value = value[0:-1]

  data['nirs'][stim].create_dataset('data', data=np.array( [np.array(start) , np.array(end) - np.array(start) , np.array(value)] ).T)

# print(start)
# print(end)
# print(start_stim)
# print(value)

  if show:
    print('Showing stim record ', stim, ' created for rest...')
    print(data['nirs'][stim]['name'][0])
    print(data['nirs'][stim]['data'][:])

  return

#@markdown `def plot_stim_stimulus(data)`
def plot_stim_stimulus(data):

  stim_list = [x for x in list(data['nirs'].keys()) if x.find('stim') == 0]
  if len(stim_list) == 0:
    print('Error: there is no stim records in data.')
    return 
    
  stim_name_set = set() 

  for s in stim_list: 
    stim_name_set.add( data['nirs'][s]['name'][0].decode('utf8') )

  stim_name_set = list(stim_name_set)

  colors = dict(zip( stim_name_set, ['g','g','b','c','m','k','y','w','r','r','b','c','m','k','y','w'] ))  

  f = plt.figure(figsize=(20,4))

  plt.hlines(0,0,channels_raw.time.max())

  for s in stim_list:
    plt.vlines(data['nirs'][s]['data'][:,0],0,data['nirs'][s]['data'][:,2],label=data['nirs'][s]['name'][0].decode('utf8'),
    color=colors[data['nirs'][s]['name'][0].decode('utf8')])

  plt.legend()
  plt.show()

  return

"""# Copia todos arquivos do drive para local"""

import os
path_in = '/content/drive/MyDrive/Cond 2/'
path_out= '/content/snirf/'

entries = os.listdir(path_in)
print(entries)

for f in entries:
  if f.count('.snirf') > 0:
    os.system('cp ' + "'" + path_in + f  + "'" + ' ' + '/content/' + f )

"""# Exclui arquivos de coleta com problema e dados diferentes de `.snrif` do diretório"""

# entries.remove('JMVR_TCC_25.snirf')
# entries.remove('JMVR_TCC_57.snirf')

delete = [] # .ipynb_checkpoints for example
for i in range(len(entries)):
  if entries[i].count('.snirf') == 0:
    delete.append(i)

# print(delete)
i = 0
for e in delete:    
  entries.pop(e - i)
  i = i + 1

entries

"""# Processa todos arquivos

**Entrada:**

> * Dados `.snirf` 

**Saída:**

> Faz as seguintes alterações criando novos arquivos `.snirf` (dentre outros, ajustes requeridos para uso no `NME-nirs`):

> * Inclui registro AppName b'snirf-Mack' para identificar os dados tratados nos arquivos
> * POSITIVA os canais com sinais < 0
> * Ajusta `time` para intervalos de tempo constante
> * Valida a carga dos dados com o `NME-nirs`
> * Cria no diretório de entrada um diretório `/snirf` com os arquivos alterados
"""

##@markdown **Process all**
for fname in entries: 
  print(100*'*')
  print('* Processing ', fname)
  print(100*'*') 

  """# Clone original `.snirf` file

  Vamos criar uma cópia do snirf original para fazer as alterações.
  """

  clone_snirf(path_in, path_out, fname=fname)

  """# Read new `.snirf`"""

  path_out='/content/snirf/'

  data = h5py.File(path_out + fname,'r+')

  if data.keys() != h5py.File(path_in + fname,'r').keys():
    print('Error: new snirf file with different keys from source')

  """# Create **Channels Table**

  Um dataframe contendo informações sobre os canais (source, detector,	wavelength,	source_pos etc.)
  """

  #
  # create alias for data['nirs']
  #
  for key in data['nirs'].keys():
    exp = key + ' = data["nirs"]["' + key + '"] '
  #  print(exp)
    exec(exp)

  channels_table = create_channels_table(data)
  # display(channels_table.head())

  """# Create **Channels Raws**

  Um dataframe contendo os sinais `raw` de cada canal.
  """

  channels_raw = create_channels_raw(data)
  # display(channels_raw.head())

  """# Plot Raw Channels to Analysis

  """

  # plotchannels_tab(channels_raw,reds=True)  
  # plotchannels_tab(channels_raw, reds=True, all=False, normalized=True, statistics=False)

  """# Corrige Valores Negativos e `sample_rate`"""

  channels_raw, rectified_negatives = rectify_negatives(channels_raw, threshold_neg = 0.10, transform='positive')

  # plotchannels_tab(channels_raw,reds=True,all=False, statistics=False, individuals=False)

  # print( list(rectified_negatives) )

  sample_rate, total_time, resample_time = check_sample_rate(channels_raw)

  channels_raw.time = resample_time

  sample_rate, total_time, resample_time = check_sample_rate(channels_raw)

  """# Ajustando o `.snirf`

  Leva as correções acima, efetuadas no dataframe, para o `.snirf`. 
  """

  add_appname_snirf(data, appname='snirf-Mack')

  # for key, item in data['nirs']['metaDataTags'].items():
  #  print(key, item[0])

  rectify_negatives_snirf(data, channels_raw, rectified_negatives)

  rectify_time_snirf(data, channels_raw)

  for c in rectified_negatives:
    # print( c, list(channels_raw.columns).index(c) - 1 )
    # print( data1['dataTimeSeries'][:, list(channels_raw.columns).index(c) - 1 ] )
    # print( any(data1['dataTimeSeries'][:, list(channels_raw.columns).index(c) - 1 ] < 0) )   
    data1['dataTimeSeries'][:, list(channels_raw.columns).index(c) - 1 ] = channels_raw[c]

  """## Start Analysis with NME"""

  # print(path_out + fname)

  raw_intensity = mne.io.read_raw_snirf(path_out + fname, verbose=True)

  raw_intensity.load_data()

  """# Dowload file"""

"""# Copy files para diretório `/snirf` no diretório original"""

if not os.path.exists(path_in + 'snirf/'):
  os.mkdir(path_in + 'snirf/')
  print("Directory " , path_in + 'snirf/' ,  "was created")
else:    
  print("Warning: directory " , path_in + 'snirf/' ,  " already exists")

entries = os.listdir(path_out)
out = path_in + 'snirf/'

for f in entries:
  os.system('cp ' + path_out + f + ' ' + "'" + out + f  + "'")

"""# Gera tabelas de scalp coupled index

"""

from mne.io import read_raw_nirx
from mne.preprocessing.nirs import (optical_density, beer_lambert_law,
                                    temporal_derivative_distribution_repair)
import mne
import pickle

entries =  os.listdir(path_out)
# entries = os.listdir(path_in + 'snirf/')
entries = ['27_run2.snirf']
print(entries)

sci_df_exists = False
sci_df = pd.DataFrame()

raw_haemo_all = {}

for fname in entries:
  raw_intensity = mne.io.read_raw_snirf(path_out + fname, verbose=True)
  # raw_intensity = mne.io.read_raw_snirf(path_in + 'snirf/' + fname, verbose=True)
  raw_intensity.load_data()
  raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
  sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
  
  if not sci_df_exists:
    sci_df['ch_names'] = raw_od.ch_names

  sci_df[fname.split('.')[0]] = sci

  raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)

  # raw_haemo = raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2,
  #                           l_trans_bandwidth=0.02)

  raw_haemo_all[fname.split('.')[0]] = raw_haemo


filename = path_in + 'raw_haemo_all.pickle'
outfile = open(filename,'wb')
pickle.dump(raw_haemo_all,outfile)
outfile.close()

sci_df.to_csv(path_in + 'sci_df.csv',index=None)

# To retrieve pickle
# infile = open(filename,'rb')
# new_dict = pickle.load(infile)
# infile.close()

display(sci_df)

raw_haemo

plt.figure(figsize=(15,15))
sns.heatmap(sci_df.set_index('ch_names'), cmap="rocket_r",alpha=0.9,square=True) # linewidths=.05,
plt.show()

"""# Channels Types: Motor x Parietal



"""

sci_df = pd.read_csv(path_in + 'sci_df.csv')
sci_df.head()

channel_types_text = """S1_D1 = Hole 1 (D1 - B2) = Motor ; 
S1_D8 = Hole 9 ( D8 - B2)= Motor ; 
S1_D14 = Hole 20 (D6 - B3) = Motor ; 
S2_D2= Hole 2(B2-D2) = Motor ; 
S2_D3= Hole 3 (B2-D3) = Motor ; 
S2_D8= Hole 9 (D8 - D2) = Motor ; 
S2_D9= Hole 13  (D1 - B3) = Motor ; 
S2_D11= Hole 16 (D3 - B3) = Parietal ; 
S3_D4= Hole 4 (D4 - B2) = Parietal ; 
S3_D5= Hole 5 (D5- B2) = Parietal ; 
S3_D9= hole 13 (D1 - B3) = Motor ; 
S3_D10= Hole 14 (D2 - B3) = Parietal ; 
S3_D12= Hole 17 (D4 - B3) = Parietal ; 
S4_D6= Hole 6 ( |D6 - B2) = Parietal ; 
S4_D7= Hole 7 (D7 - B2) = Parietal ; 
S4_D10= Hole 14 (D2 - B3) = Parietal ; 
S4_D13= Hole 19 (D5 - B3) = Parietal ; 
S5_D8= Hole 9 (D8 - B1) = Motor ; 
S5_D11= Hole  16 (D3 - B3) = Parietal ; 
S5_D14= Hole 20 (D6 - B3) = Motor ; 
S5_D15= Hole 21 (D7 - B3) = Motor ; 
S6_D10= Hole 14 (D2 - B3) = Parietal ;
S6_D12= Hole 17 (D4 - B3) = Parietal ; 
S6_D13= Hole 19 (D5 - B3) = Parietal"""

import pandas as pd

channel_SD = pd.DataFrame()
channel_name = []
channel_type = []
channel_hole = []

for ch in channel_types_text.replace(' ','').replace('\n','').split(';'):
  channel_name.append( ch.split('=')[0] )
  channel_type.append( ch.split('=')[2] )
  channel_hole.append( ch.split('=')[1] )

channel_SD['channel_name'] = channel_name
channel_SD['channel_type'] = channel_type
channel_SD['channel_hole'] = channel_hole

channel_SD

list( channel_SD[channel_SD.channel_type == 'Motor' ].channel_name )

temp = pd.DataFrame({'sd':sci_df.ch_names.apply(lambda x: x.split(' ')[0])}).merge(channel_SD, left_on='sd', right_on='channel_name')
sci_df['channel_type'] = temp['channel_type']
sci_df.head()

plt.figure(figsize=(15,15))
sci_motor = sci_df[ sci_df.channel_type == 'Motor' ].set_index('ch_names').drop(columns='channel_type')
sns.heatmap(sci_motor > 0.85, cmap="rocket_r",alpha=0.9,square=True) # linewidths=.05,
plt.title('SCI Motor Channels',fontsize=16)
plt.show()

plt.figure(figsize=(15,15))
sci_parietal = sci_df[ sci_df.channel_type == 'Parietal' ].set_index('ch_names').drop(columns='channel_type')
sns.heatmap(sci_parietal > 0.85, cmap="rocket_r",alpha=0.9,square=True) # linewidths=.05,
plt.title('SCI Parietal Channels',fontsize=16)
plt.show()

"""# Retrieve one subject"""

from mne.io import read_raw_nirx
from mne.preprocessing.nirs import (optical_density, beer_lambert_law,
                                    temporal_derivative_distribution_repair)
import mne
import pickle

subject = '21_run2'
threshold = 0.95

entries = [subject + '.snirf']
print(entries)

sci_df_exists = False
sci_df = pd.DataFrame()

raw_haemo_all = {}

for fname in entries:
  raw_intensity = mne.io.read_raw_snirf(path_in + 'snirf/' + fname, verbose=True)
  raw_intensity.load_data()
  raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
  sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
  
  fig, ax = plt.subplots()
  ax.hist(sci)
  ax.set(xlabel='Scalp Coupling Index', ylabel='Count', xlim=[0, 1])
  plt.show()
  
  raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < threshold))
  raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)

  fig = raw_haemo.plot_psd(average=True)
  fig.suptitle('Before filtering', weight='bold', size='x-large')
  fig.subplots_adjust(top=0.88)
  raw_haemo = raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2,
                              l_trans_bandwidth=0.02)
  fig = raw_haemo.plot_psd(average=True)
  fig.suptitle('After filtering', weight='bold', size='x-large')
  fig.subplots_adjust(top=0.88)
  plt.show()

  print(raw_haemo.info)

  raw_haemo.plot(n_channels=48, duration=1000, show_scrollbars=False)
  plt.show()

reject_criteria = dict(hbo=80e-6)
tmin, tmax = -10, 18

epochs = mne.Epochs(raw_haemo, events, event_id=event_dict,
                    tmin=tmin, tmax=tmax,
#                    reject=reject_criteria, reject_by_annotation=True,
                    proj=True, baseline=(None, 0), preload=True,
                    detrend=None, verbose=True,
                    event_repeated='merge') # ADICIONADO Allowed values are 'error', 'drop', and 'merge'
# epochs.plot_drop_log()

epochs['controle']

styles_dict

evoked_dict = {'Sync/HbO': epochs['Sync'].average(picks='hbo'),
               'Sync/HbR': epochs['Sync'].average(picks='hbr'),
               'controle/HbO': epochs['controle'].average(picks='hbo'),
               'controle/HbR': epochs['controle'].average(picks='hbr')}

# Rename channels until the encoding of frequency in ch_name is fixed
for condition in evoked_dict:
    evoked_dict[condition].rename_channels(lambda x: x[:-4])

color_dict = dict(HbO='#AA3377', HbR='b')
styles_dict = dict(controle=dict(linestyle='dashed'))

mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.95,
                             colors=color_dict, styles=styles_dict)

evoked_dict = {'Async/HbO': epochs['Async'].average(picks='hbo'),
               'Async/HbR': epochs['Async'].average(picks='hbr'),
               'controle/HbO': epochs['controle'].average(picks='hbo'),
               'controle/HbR': epochs['controle'].average(picks='hbr')}

# Rename channels until the encoding of frequency in ch_name is fixed
for condition in evoked_dict:
    evoked_dict[condition].rename_channels(lambda x: x[:-4])

color_dict = dict(HbO='#AA3377', HbR='b')
styles_dict = dict(controle=dict(linestyle='dashed'))

mne.viz.plot_compare_evokeds(evoked_dict, combine="mean", ci=0.95,
                             colors=color_dict, styles=styles_dict)

import pickle

# To retrieve pickle
filename = path_in + 'raw_haemo_all.pickle'
infile = open(filename,'rb')
raw_haemo_all = pickle.load(infile)
infile.close()

raw_haemo_all.keys()

subject = '27_run2'
raw_haemo = raw_haemo_all[subject]

from mne.io import read_raw_nirx
from mne.preprocessing.nirs import (optical_density, beer_lambert_law,
                                    temporal_derivative_distribution_repair)
import mne

fig = raw_haemo.plot_psd(average=True)
fig.suptitle('Before filtering', weight='bold', size='x-large')
fig.subplots_adjust(top=0.88)

raw_haemo = raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2,
                             l_trans_bandwidth=0.02)

fig = raw_haemo.plot_psd(average=True)
fig.suptitle('After filtering', weight='bold', size='x-large')
fig.subplots_adjust(top=0.88)
plt.show()

raw_haemo.info

list( sci_df[ sci_df[subject] < 0.75 ].ch_names )

raw_haemo.info['bads'] = list( sci_df[ sci_df[subject] < 0.75 ].ch_names )

raw_haemo.info

events, _ = mne.events_from_annotations(raw_haemo)

event_dict = {'Async': 2, 'Sync': 3, 'controle': 4}

fig = mne.viz.plot_events(events, event_id=event_dict,
                          sfreq=raw_haemo.info['sfreq'])
fig.subplots_adjust(right=0.7)  # make room for the legend

reject_criteria = dict(hbo=80e-6)
tmin, tmax = -5, 15

epochs = mne.Epochs(raw_haemo, events, event_id=event_dict,
                    tmin=tmin, tmax=tmax,
                    reject=reject_criteria, reject_by_annotation=True,
                    proj=True, baseline=(None, 0), preload=True,
                    detrend=None, verbose=True,
                    event_repeated='merge') # ADICIONADO Allowed values are 'error', 'drop', and 'merge'
# epochs.plot_drop_log()

hbo_hbr = {}

for c in ['Async', 'Sync', 'controle']:
  df_hbo = epochs[c].average(picks='hbo').to_data_frame()
  df_hbo = pd.melt(df_hbo, id_vars=['time'])
  df_hbr = epochs[c].average(picks='hbr').to_data_frame()
  df_hbr = pd.melt(df_hbr, id_vars=['time'])
  hbo_hbr[c] = {}
  hbo_hbr[c]['hbo'] = df_hbo
  hbo_hbr[c]['hbr'] = df_hbr

for c in ['Async', 'Sync', 'controle']:

  plt.figure(figsize=(10,8))
  sns.lineplot(x=hbo_hbr[c]['hbo'].time, y=hbo_hbr[c]['hbo'].value,color='b',label= c + ' HbO',ci=99)
  sns.lineplot(x=hbo_hbr[c]['hbr'].time, y=hbo_hbr[c]['hbr'].value,color='r',label= c + ' HbR',ci=99)

  plt.legend()
  plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
clims = dict(hbo=[-20, 20], hbr=[-20, 20])
epochs['Async'].average().plot_image(axes=axes[:, 0], clim=clims)
epochs['Sync'].average().plot_image(axes=axes[:, 1], clim=clims)
for column, condition in enumerate(['Async', 'Sync']):
    for ax in axes[:, column]:
      ax.set_title('{}: {}'.format(condition, ax.get_title()))

"""# Plot all `.snirf` created"""

entries = os.listdir(out)

first = True

for fname in entries:
  data = h5py.File(out + fname,'r+')
  #
  # create alias for data['nirs']
  #
  for key in data['nirs'].keys():
    exp = key + ' = data["nirs"]["' + key + '"] '
  #  print(exp)
    exec(exp)

  channels_table = create_channels_table(data)
  channels_raw = create_channels_raw(data)

  print(100*'*')
  print('* Plotting ', fname)
  print(100*'*') 
  print()
  print()
  
  if first:
    plotchannels_tab(channels_raw,reds=True,statistics=True) 
    first = False
  else:  
    plotchannels_tab(channels_raw, reds=True, all=False, statistics=True, individuals=False)

  print()
  print()
  print('* End plotting ', fname)
  print(100*'*')

"""# Opcional, download files"""

from google.colab import files

download = input('Deseja fazer download dos arquivos criados? (y|n)')

if download == 'y':
  entries = os.listdir(path_out)
  for f in entries:
    files.download(path_out + f)