# Naming Convention
Each folder of the format Best_NN_V* is a manual blend of predictions from the same model at 3 different epochs. "Blend_NN_1_2_3__4__5" Basically means that Best_NN_v1, Best_NN_v2 and Best_NN_v3 have been blended together to yield a single probability file which has then been blended with Best_NN_v4 and Best_NN_v5 to yield another, single probability file. Each addition of "\_" signifies an increase in the blend depth.

# Whats up with blending on different epochs?
The epochs should preferably be chosen with an interval of two or more in order to keep the correlation at a minimum. The logic behind this is that, due to the learning_rate, the weights in the next epoch might not change much so the predictions might remain similar.
