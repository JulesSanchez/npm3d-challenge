import numpy as np 

def get_even_number(data,label,size=1000):
    #We assume size > n_label for each label
    list_labels = np.unique(label)
    new_data = []
    new_label = []
    for k in range(len(list_labels)):
        local_label = list_labels[k]
        if local_label ==0:
            continue
        indices = label == local_label
        data_local = data[indices]
        new_indices = np.random.choice(np.arange(len(data_local)),size,replace=False)
        new_data.append(data_local[new_indices])
        new_label += [local_label]*size
    return np.array(new_data).reshape(-1,3), np.array(new_label)
        