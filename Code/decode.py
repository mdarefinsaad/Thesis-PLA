import numpy as np


# load npz file of one scenario from path
def npz(path):
    file = np.load(path, allow_pickle=False)
    header, data = file['header'], file['data']
    return np.swapaxes(data['cirs'], 0, 1)


# get scenario name for file name
def scenario(file_name):
    if 'reflector' in file_name:
        scenario_name = 'asymm_reflector'
    elif 'varspeed' in file_name:
        scenario_name = 'symm_varspeed'
    elif 'asymm' in file_name:
        if 'nomove' in file_name:
            scenario_name = 'asymm_nomove'
        else:
            scenario_name = 'asymm'
    elif 'symm' in file_name:
        if 'nomove' in file_name:
            scenario_name = 'symm_nomove'
        else:
            scenario_name = 'symm'
    elif 'all' in file_name:
        scenario_name = 'all'
    else:
        scenario_name = 'undefined'
    return scenario_name


# get anchor point name for channel number
def anchor(channel_number, abe=False):
    if abe:
        if channel_number == 0:
            a_name = 'B -> A'
        elif channel_number == 1:
            a_name = 'B -> E'
        elif channel_number == 3:
            a_name = 'A -> B'
        elif channel_number == 4:
            a_name = 'A -> E'
        else:
            a_name = 'undefined'
    else:
        if channel_number == 0:
            a_name = 't0 -> a0'
        elif channel_number == 1:
            a_name = 't0 -> a1'
        elif channel_number == 2:
            a_name = 't0 -> a2'
        elif channel_number == 3:
            a_name = 'a0 -> t0'
        elif channel_number == 6:
            a_name = 'a1 -> t0'
        elif channel_number == 9:
            a_name = 'a2 -> t0'
        else:
            a_name = 'undefined'
    return a_name


# return full name of metric
def metric_label(metric_name):
    if metric_name == 'correlation_coeff' or metric_name == 'corr_coeff':
        label = 'Correlation Coefficient'
    elif metric_name == 'BER':
        label = 'Bit Error Ratio'
    elif metric_name == 'entropy':
        label = 'Entropy'
    elif metric_name == 'mutual information' or metric_name == 'mutinf':
        label = 'Mutual Information'
    elif metric_name == 'RMSE':
        label = 'Root Mean Square Error'
    elif metric_name == 'variance':
        label = 'Joint Performance Metric'
    else:
        label = metric_name
    return label
