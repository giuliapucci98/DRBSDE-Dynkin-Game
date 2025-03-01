# Solving Dynkin Games via Doubly Reflected BSDEs

This repository contains code for solving Doubly Reflected Backward Stochastic Differential Equations (DRBSDEs) and implementing Dynkin games.  

## Code Structure  

•⁠  **⁠DRBSDE.py**: Contains the solver and necessary classes for solving a DRBSDE using a backward-in-time algorithm

•⁠  **⁠DynkinGame.py**: Main file that should be run. It implements the Dynkin game and includes an analysis of exit times, which can be performed when solving the game multiple independent times.  

## Results
In folders named **Test_Dynkin** and **Test_Dynkin_CfD**, we keep the graphical and computational results obtained from the simulations used in the paper. You can find figures in the subfolder **Graphs**, while in the subfolder **state_dicts**, we keep the neural network's trained weights. 

## Requirements  

This project is implemented in Python 3.9 and uses PyTorch 2.5.1
