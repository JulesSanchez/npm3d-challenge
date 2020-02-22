import numpy as np 

def get_even_number(data,label,size=1000,return_indices=False):
    #We assume size > n_label for each label
    list_labels, counts = np.unique(label, return_counts=True)
    size = min(np.min(counts),size)
    new_data = []
    new_label = []
    ind = []
    for k in range(len(list_labels)):
        local_label = list_labels[k]
        if local_label ==0:
            continue
        indices = label == local_label
        new_indices = np.random.choice(np.arange(len(label[indices])),size,replace=False)
        ind.append(np.where(indices)[0][new_indices])
        new_label += [local_label]*size
        if not return_indices:
            data_local = data[indices]
            new_data.append(data_local[new_indices])
    if return_indices:
        return np.array(ind).reshape(-1), np.array(new_label) 
    return np.array(new_data).reshape(-1,3), np.array(new_label)
