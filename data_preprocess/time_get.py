import os
import csv
import datetime
import wave

# Function to extract event start times and durations for further audio segmentation
def time_information(txt_path, patient_name, save_path, start_time, directory):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    event_times = []
    event_durations = []

    # Open the WAV file to calculate its total duration
    with wave.open(directory + patient_name + '.WAV', "r") as audio:
        num_frames = audio.getnframes()
        frame_rate = audio.getframerate()
    duration = int(num_frames / float(frame_rate))

    # Read the text file containing timestamps and durations
    with open(txt_path) as f:
        lines = f.readlines()

    # Process each line to extract event times and durations for MBR audio
    for line in lines:
        cols = line.strip().split("\t")
        if len(cols) == 6:
            event_time = cols[2]  # Extract the event time
            event_duration = cols[4]  # Extract the event duration
        elif len(cols) == 5:
            event_time = cols[1]
            event_duration = cols[3]
        elif len(cols) == 4:
            event_time = cols[0]
            event_duration = cols[2]
        else:
            print("Error in txt file, please check!")

        # Convert time string to seconds from the start time
        if ":" in event_time:
            event_start_time = datetime.datetime.strptime(event_time, "%H:%M:%S")
            if event_start_time < start_time:
                event_start_time += datetime.timedelta(days=1)
            event_times.append((event_start_time - start_time).total_seconds())
            event_durations.append(int(event_duration))

    # Save the event times and durations to a CSV file
    with open(save_path + patient_name + "-events.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "duration"])
        for i in range(len(event_times)):
            writer.writerow([event_times[i], event_durations[i]])

    print("Audio categorization complete!")
