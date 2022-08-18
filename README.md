# Intelligent Control of Cardiac Rhythms using Artificial Neural Networks

Gabriel S. Lima, gabriel.lima.095@ufrn.edu.br, 
Wallace M. Bessa, wmobes@utu.fi,
Marcelo A. Savi, savi@mecanica.coppe.ufrj.br

In this project you can find the:
(a) codes used for implementating the intelligent controller for cardiac rhythms;
(b) and all the results obtained by this implementation considering the intelligent and conventional approaches.

The codes are separated in two parts:
(a) a main file with the definition of all necessary variables;
(b) a header file with all the necessary functions.

The columns of the results file are divided as follows:
[1] dimensional time;
[2] ECG signal;
[3] derivative of the ECG signal;
[4] control action;
[5] desired ECG;
[6] derivative of the desired ECG;	
[7] tracking error;
[8] derivative of the tracking error;
[9] combined error signal;
[10] uncertainty estimation;
[11] norm of the ANN weights;
[12] SA node signal;
[13] AV node signal;
[14] HP node signal;
[15] derivative of the SA node signal;
[16] derivative of the AV node signal;
[17] derivative of the HP node signal.

The Armadillo library [1] is needed for the implementation.

[1] Sanderson, C. and Curtin, R., 2016. Armadillo: a template-based C++ library for linear algebra. Journal of Open Source Software, 1(2), p.26.
