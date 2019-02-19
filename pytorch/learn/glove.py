# Fast implemenetation : https://gist.github.com/MatthieuBizien/de26a7a2663f00ca16d8d2558815e9a6
# normal implementation : https://github.com/2014mchidamb/TorchGlove/blob/master/glove.py


import numpy as np

cooccurance_matrix = np.random.randint(0, high=10, size=(3,3))


# initialize vectores
word_vector_size = 10
vocab_size = len(cooccurance_matrix)

# random initialization of vectors
word_vectors = np.random.random((vocab_size, word_vector_size))
word_bias = word_vectors[0]
context_bias = word_vectors[1]

focus_word=0
context_word=1

def get_batch():
    loss = np.matmul(focus_word, context_word) + word_bias + context_bias - math.log(cooccurance_matrix[focus_word])
    loss = compute_weight(cooccurance_matrix[focus_word, context_word])
    total_loss = sum_over (loss)


    return (focus_word, context_word, cooccurance_matrix[focus_word, context_word], cooccurance_matrix[focus_word])
