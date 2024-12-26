# DFNet
This is repo of paper 'An End-to-End Audio Classification Framework with Diverse Features for Obstructive Sleep Apnea-Hypopnea Syndrome Diagnosis'.


## Data Preparation File Format

Each annotation file should follow this format:
```bash
Posture  Timestamp  Event_Type  Duration  Recording
```
The mean of each line:

**Field Descriptions:**
- **Posture**: The physical orientation of the body during the recording, such as supine (lying on the back), prone (lying on the stomach), or lateral (lying on the side).
- **Timestamp**: The time at which the event was recorded, formatted as HH:MM:SS.
- **Event_Type**: The type of event that is being recorded, such as snoring or a breathing pause.
- **Duration**: The duration of the event, typically measured in seconds.
- **Recording**: Additional notes or data about the recording.

## Extract MFCC
Run the Preprocessing Script: Use the `data_proprocess/run.py` script. This script processes each audio file in the specified directory, performing tasks like time information extraction, audio cutting, and MFCC feature extraction.

## Train & Evaluate

Run the script `main.py` train DFNet and calculate the accuracy and F1 score.

```bash
python main.py
```

## Acknowledgments and Citations

### Software and Libraries
We gratefully acknowledge the following resources and contributions to our work:

- **Python Speech Features** by James Lyons. For more information and to cite the software, please refer to:
  - Lyons, James et al. (2020, January 14). _jameslyons/python_speech_features: release v0.6.1_ (Version 0.6.1). Zenodo. [DOI: 10.5281/zenodo.3607820](http://doi.org/10.5281/zenodo.3607820)

### Research Papers
For theoretical foundations and methodology, the following papers were instrumental to our research:

- **ROCKET: Exceptionally Fast and Accurate Time Series Classification Using Random Convolutional Kernels**
  - Dempster, Angus, Petitjean, François, and Webb, Geoffrey I. _Data Mining and Knowledge Discovery_, 34(5), 1454-1495, 2020.
  
- **MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification**
  - Dempster, Angus, Schmidt, Daniel F, and Webb, Geoffrey I. _Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining_. ACM, New York, 2021, pp. 248–257.
