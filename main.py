import numpy as np
import matplotlib.pyplot as plt
from helper import *

def main():

    #  IMPORTING DATA
    X_train, y_train, X_test, y_test = get_training_testing_data('train_catvnoncat.h5', 'test_catvnoncat.h5')

    #  PARAMETERS
    n_x = X_train.shape[0]  # num_px * num_px * 3
    n_h = 7  #   number of hidden layers
    n_y = 1  # number of output
    layers_dims = (n_x, n_h, n_y)

    # CALLING THE OUTPUTS OF THE MODEL
    parameters = L_layer_model(X_train, y_train, layers_dims, num_iterations=2000, print_cost=True)

    # PREDICTIONS
    pred_train = predict(parameters, X_train)
    pred_test = predict(parameters, X_test)
    score_test = 1 - np.mean(abs(pred_test - y_test))
    score_train = 1 - np.mean(abs(pred_train - y_train))
    print("\n\nTraining set accuracy is " + str(np.round(score_train, 2)))
    print("Testing set accuracy is " + str(np.round(score_test, 2)))
    plt.show()



if __name__ == '__main__':
    main()