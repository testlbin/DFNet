import os
import datetime
import numpy as np
from cut_wav import cut_wav
from time_get import time_information
from mfcc import get_MFCC


directory = 'your_audio_file_path'

# Your audio recording start time
start_time = datetime.datetime.strptime("23:00:05", "%H:%M:%S")

# auto label data
for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        print('------------------------------------')
        file_name, file_ext = os.path.splitext(filename)
        txt_path = directory + 'txt/' + file_name + '-label.txt'
        csv_path = directory + 'event_record/'
        mfcc_path = directory + 'MFCC/'
        cut_wav_path = directory + 'cut_audio/' + file_name + '/'

        duration = directory + file_name + '.WAV'

        time_information(txt_path, file_name, csv_path, start_time, directory)
        cut_wav(file_name, directory)
        get_MFCC(directory, file_name, cut_wav_path, mfcc_path)