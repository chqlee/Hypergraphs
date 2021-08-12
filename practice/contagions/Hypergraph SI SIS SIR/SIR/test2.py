import numpy as np

list = [1,2,3,4,5]
list2 = [1,2]
list.remove(list2)
np_list = np.array(list)
np_list2 = np.array(list2)
print(np.delete(np_list,2))