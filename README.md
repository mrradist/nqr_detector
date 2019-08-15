The TNT detector file contains an implementation of various methods for detecting an NQR signal with a priori unknown parameters. Methods
1) The method of neural network (NN). The neural network for detection is described in the file.
2) The method of multiple consistent filters (SMF).  Filters for the method are formed in the file. The detection method is contained in the explose_det function.
3) Energy detection method with filtration (ED). The method is also contained in the explose_det function.
4) Optimal method. This method knows the parameters of the desired signal and is used only to estimate the maximum accuracy of detection.

As you can see, the accuracy of detection by non-neural network methods is determined in the function explose_det in the file function.py. The results of this function: p_e is the probability of detecting the NQR signal of the ED methods, p is the probability of detection by the optimal methods, which knows the parameters of the signal, p_m is the probability of detection by the SMF.

The function.py file also contains functions for loading signals from the dataset (get_batch) and posting ROC curves.

The dataset_generator_TNT.py file allows you to create TNT model NQR signals and generate datasets from them. In this case, the ranges of uncertainties and various parameters of the signals can be changed.
