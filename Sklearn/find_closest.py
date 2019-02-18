import numpy as np
from sklearn.neighbors import NearestNeighbors
import re
from scipy.spatial import distance

# Read responses
responses = list()
with open(r"G:\Data\Skype\SV02\ResponseSetUNK\filtered_response_set.txt", "r", encoding="utf-8") as f_in:
    for line in f_in:
        fields = re.split("\t", line.strip())
        responses.append(fields[2])

# Read response vectors
response_vector = np.genfromtxt(r"G:\Data\Skype\SV02\DebugUNK\response_vec\vectors.txt", delimiter=r' ')

neigh = NearestNeighbors(3, 0.4)
neigh.fit(response_vector)

#x, y = neigh.kneighbors([response_vector[0]], 50, return_distance=True)

#for idx, id in enumerate(y[0]):
#    print ("{}\t{}".format(responses[id], x[0][idx]))

# compute cosine distance and educledian distance
for idx, vec in enumerate(response_vector):
    print("{}\t{}".format(responses[idx], distance.cosine(response_vector[0], response_vector[idx])))
    if idx>100:
        break
