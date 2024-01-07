# A Personal Health Agent for Decision Support in Arrhythmia Diagnosis

This repository contains work published by Wanyana et al. (2023).

We propose an architecture for a personal health agent (PHA) that combines machine learning (ML) and a Bayesian network for detecting and diagnosing heart disease, specifically arrhythmia. This repository contains the code for the ML component of the PHA, which classifies a person’s ECG signal. Four ML models, i.e. gradient boosting, random forest, multilayer perceptron and support vector machine, are compared and evaluated using a dataset of 5,340 records containing 12-lead ECG signals created from the Chapman-Shaoxing database. Among the four models, the gradient boosting model produces the best accuracy of 82.88% when classifying an ECG signal as either atrial fibrillation, other arrhythmia, or no arrhythmia. The detected pattern is integrated into a BN that captures expert knowledge about the causes of arrhythmia.

## Requirements
This project uses Python 3.10. For other requirements see `requirements.txt`. The use of a virtual environment is strongly recommended.

## Data
Create a `data` directory where the ECG data will be stored. This project uses the Chapman-Shaoxing dataset (Zheng et al., 2020), which can be downloaded from the [Physionet The PhysioNet/Computing in Cardiology Challenge 2021 files](https://physionet.org/content/challenge-2021/1.0.3/). Download the files using your terminal:

`wget -r -N -c -np https://physionet.org/files/challenge-2021/1.0.3/`

## References
Wanyana, T., Nzomo, M., Price, C.S., Moodley, D. (2023). A Personal Health Agent for Decision Support in Arrhythmia Diagnosis. In: Maciaszek, L.A., Mulvenna, M.D., Ziefle, M. (eds) Information and Communication Technologies for Ageing Well and e-Health. ICT4AWE ICT4AWE 2021 2022. Communications in Computer and Information Science, vol 1856. Springer, Cham. https://doi.org/10.1007/978-3-031-37496-8_20

Zheng, J., Zhang, J., Danioko, S. et al. A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. Sci Data 7, 48 (2020). https://doi.org/10.1038/s41597-020-0386-x
