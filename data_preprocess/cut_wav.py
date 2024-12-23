import wave
import os
import time
from data_processing.pre_processing import data_process
from data_processing.pre_processing import Ndata_process

def write_wav(path,data,wav_chn,wav_sampwidth,wav_samprate):
    f = wave.open(path, 'wb')
    f.setnchannels(wav_chn)
    f.setsampwidth(wav_sampwidth)
    f.setframerate(wav_samprate)
    f.writeframes(data)
    f.close()

"""
cut audio
"""

def cut_wav(patient_name, dir_path):
    # Start the timer for the program execution
    time_start = time.time()
    print('Cutting audio:', patient_name)
    # Retrieve the audio data
    path_wav = dir_path + patient_name + '.wav'
    path_data1 = dir_path + 'event_record/' + patient_name + '-event_audio.csv'
    path_data2 = dir_path + 'event_record/' + patient_name + '-normal_audio.csv'
    save_path = dir_path + 'cut_audio/' + patient_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    wf = wave.open(os.path.join(path_wav), 'rb')

    # Retrieve audio properties
    wav_chn = wf.getnchannels()  # channels
    wav_sampwidth = wf.getsampwidth()  # sample width in bytes per frame
    wav_samprate = wf.getframerate()  # frame rate: number of frames per second
    wav_framenumber = wf.getnframes()  # total number of frames

    # Read the entire audio data
    data = wf.readframes(wav_framenumber)
    wf.close()

    # Calculate the segments of audio to be cut
    start_time, length_of_time = data_process(path_data1)

    # Cut and save audio for events
    for i in range(0, start_time.shape[1]):
        start_point = start_time[:, i] * 1000
        end_point = start_point + (length_of_time[:, i]) * 1000
        buff = data[int(start_point) * 8 * 2: int(end_point) * 8 * 2]  # Slice the binary string
        write_wav((save_path + 'snore_' + '{}'.format(i) + '.wav'), buff, wav_chn, wav_sampwidth, wav_samprate)

    # Cut and save audio for non-events
    start_time_, length_of_time_, end_ = Ndata_process(path_data2)

    for i in range(0, start_time_.shape[1]):
        if 60 >= length_of_time_[:, i] >= 10:
            start_point = start_time_[:, i] * 1000
            end_point = start_point + (length_of_time_[:, i]) * 1000
            buff = data[int(start_point) * 8 * 2: int(end_point) * 8 * 2]  # Slice the binary string
            write_wav((save_path + 'normal_' + '{}'.format(i) + '.wav'), buff, wav_chn, wav_sampwidth, wav_samprate)
        elif length_of_time_[:, i] >= 60:
            CutframeNum = wav_samprate * 60  # Frames for 60 seconds
            CutNum = (length_of_time_[:, i] * wav_samprate) / CutframeNum
            if CutNum >= 7:
                pass
            else:
                for j in range(int(CutNum)):
                    start_point = (start_time_[:, i] + 60 * j) * 1000
                    end_point = start_point + 60 * 1000
                    if end_point <= (end_[:, i]) * 1000 and start_point - end_point <= 60:
                        end_time = start_point + 60 * 1000
                        buff = data[int(start_point) * 8 * 2: int(end_time) * 8 * 2]  # Slice the binary string
                        write_wav((save_path + 'normal_' + '{}'.format(i) + '-{}'.format(j) + '.wav'), buff,
                                  wav_chn, wav_sampwidth, wav_samprate)
                    else:
                        end_point = end_[:, i] * 1000
                        buff = data[int(start_point) * 8 * 2: int(end_point) * 8 * 2]
                        write_wav((save_path + 'normal_' + '{}'.format(i) + '-{}'.format(j) + '.wav'), buff,
                                  wav_chn, wav_sampwidth, wav_samprate)
        else:
            pass

    time_end = time.time()
    time_sum = time_end - time_start
    print("Time taken to cut audio: %f seconds" % time_sum)
    print('Cutting complete -', patient_name)




