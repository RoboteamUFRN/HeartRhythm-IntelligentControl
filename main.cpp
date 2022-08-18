#define _USE_MATH_DEFINES // for C++
#define ARMA_USE_CXX11

#include <cstdlib>
#include <math.h>
#include <random>
#include <cmath>
#include <iostream>
#include <time.h>
#include <fstream>
#include <stdio.h>
#include <chrono>
#include <armadillo>
using namespace std;
using namespace arma;
//#include "normal_rhythm.h"
//#include "atrial_flutter.h"
//#include "atrial_fibrillation.h"
#include "ventricular_flutter.h"
//#include "ventricular_fibrillation_wStim.h"
//#include "ventricular_fibrillation_wouStim.h"

double sinal(double x);

int main(int argc, char** argv)
{

	errno_t saida, sw;
	FILE* saida1, *saidaw;
    saida = fopen_s(&saida1, "results.txt", "w");
	if (!saida1)
	{
		perror("File opening failed");
		return EXIT_FAILURE;
	}
	sw = fopen_s(&saidaw, "weights.txt", "w");
	if (!saidaw)
	{
		perror("File opening failed");
		return EXIT_FAILURE;
	}

	double t = 0.0;
	double tf = 399.99;
	double csr = 100.0;
	double cst = 1.0 / csr;
	double ssr = 1000.0;
	double sst = 1.0 / ssr;
	double asr = 10.0;
	double ast = 0.0;
	double stf, atf = 0.0;

	arma::vec Q(3), Qp(3), Qpp(3, fill::zeros), Q_old(3);
	arma::vec Q_delay(6);

	double ECG, ECGp, ECGpp;

	Q(0) = -0.1;
	Qp(0) = 0.025;
	Q(1) = -0.6;
	Qp(1) = 0.1;
	Q(2) = -3.3;
	Qp(2) = 10.0/15.0;
	Q_old.fill(0.0);

	ECG = 1.0 + 0.06 * Q(0) + 0.1 * Q(1) + 0.3 * Q(2);
	ECGp = 0.06 * Qp(0) + 0.1 * Qp(1) + 0.3 * Qp(2);

	xt1_SA_AV_old = Q(0);
	xt1_SA_HP_old = Q(0);
	xt3_AV_SA_old = Q(1);
	xt3_AV_HP_old = Q(1);
	xt5_HP_SA_old = Q(2);
	xt5_HP_AV_old = Q(2);

	Q1d_SA_HP.set_size(tau_SA_HP / sst); Q1d_SA_HP.fill(0.0);
	Q1d_SA_AV.set_size(tau_SA_AV / sst); Q1d_SA_AV.fill(0.0);
	Q3d_AV_HP.set_size(tau_AV_HP / sst); Q3d_AV_HP.fill(0.0);
	Q3d_AV_SA.set_size(tau_AV_SA / sst); Q3d_AV_HP.fill(0.0);
	Q5d_HP_AV.set_size(tau_HP_AV / sst); Q5d_HP_AV.fill(0.0);
	Q5d_HP_SA.set_size(tau_HP_SA / sst); Q5d_HP_SA.fill(0.0);
	delay = { tau_AV_SA, tau_HP_SA, tau_SA_AV, tau_HP_AV, tau_SA_HP, tau_AV_HP };
	delay_size = { tau_AV_SA / sst, tau_HP_SA / sst, tau_SA_AV / sst, tau_HP_AV / sst, tau_SA_HP / sst, tau_AV_HP / sst };

	cout << delay_size(2) << endl;

	ifstream file_in("upo_csr100.txt");
	if (!file_in.is_open()) {
		cout << "Error!\n";
		return 0;
	}

	double u = 0.0, xd, xpd, xppd, e, ep, s;
	double lambda = 10.0;
	double d_est = 0.0;
	double u_max = 3000.0;
	double kappa = 15.0;
	double b_est = 1.0;

	double learn = 100.0;
	int contador_max = 500000000;
	int contador = 1;
	double cam = 20.0;
	int n = 6;
	vec weight(n), width(n), center(n), V_ativ(n);
	weight.fill(0.0);
	center = { -cam / 2.0, -cam / 8.0, -cam / 16.0, cam / 16.0, cam / 8.0, cam / 2.0 };
	width = { cam / 2.0, cam / 3.0, cam / 6.0, cam / 6.0, cam / 3.0, cam / 2.0 };
	double lim_weight = 3000.0, lim_s = 1.0;
	int training_opt = 0;

	while (t <= tf){

		file_in >> xd;
		file_in >> xpd;
		file_in >> xppd;

		e = ECG - xd;
		ep = ECGp - xpd;

		s = ep + lambda * e;

		if (t * 0.1048 < 16.0) {
			u = 0.0;
		}
		else {
			ativ(s, center, width, n, V_ativ);
			atual(t, t + cst, weight, center, width, learn, s, contador_max, &contador, V_ativ, lim_weight, lim_s, training_opt);
			d_est = dot(V_ativ, weight);
		
			u = (xppd - 2.0 * lambda * ep - lambda * lambda * e - d_est);
		}

		if (fabs(u) > u_max) { u = u_max * sinal(u); }

		stf = t + cst;
        while (t < stf) {
			if (delay_size(0) < 1.0) {
				Q_delay(0) = Q(1);
			}
			else {
				if (t < delay(0)) {
					Q_delay(0) = Q_old(1) - delay(0) * (Q(1) - Q_old(1)) / sst;
					Q3d_AV_SA(cont_delay(0)) = Q_delay(0);
					cont_delay(0) += 1.0;
				}
				else {
					Q_delay(0) = Q3d_AV_SA(0);
					for (int i = 0; i < cont_delay(0) - 1; i++) {
						Q3d_AV_SA(i) = Q3d_AV_SA(i + 1);
					}
					Q3d_AV_SA(cont_delay(0) - 1) = Q(1);
				}
			}
			
			if (delay_size(1) < 1.0) {
				Q_delay(1) = Q(2);
			}
			else {
				if (t < delay(1)) {
					Q_delay(1) = Q_old(2) - delay(1) * (Q(2) - Q_old(2)) / sst;;
					Q5d_HP_SA(cont_delay(1)) = Q_delay(1);
					cont_delay(1) += 1.0;
				}
				else {
					Q_delay(1) = Q5d_HP_SA(0);
					for (int i = 0; i < cont_delay(1) - 1; i++) {
						Q5d_HP_SA(i) = Q5d_HP_SA(i + 1);
					}
					Q5d_HP_SA(cont_delay(1) - 1) = Q(2);
				}
			}
			
			if (delay_size(2) < 1.0) {
				Q_delay(2) = Q(0);
			}
			else {
				if (t < delay(2)) {
					Q_delay(2) = Q_old(0) - delay(2) * (Q(0) - Q_old(0)) / sst;
					Q1d_SA_AV(cont_delay(2)) = Q_delay(2);
					cont_delay(2) += 1.0;
				}
				else {
					Q_delay(2) = Q1d_SA_AV(0);
					for (int i = 0; i < cont_delay(2) - 1; i++) {
						Q1d_SA_AV(i) = Q1d_SA_AV(i + 1);
					}
					Q1d_SA_AV(cont_delay(2) - 1) = Q(0);
				}
			}

			if (delay_size(3) < 1.0) {
				Q_delay(3) = Q(2);
			}
			else {
				if (t < delay(3)) {
					Q_delay(3) = Q_old(2) - delay(3) * (Q(2) - Q_old(2)) / sst;
					Q5d_HP_AV(cont_delay(3)) = Q_delay(3);
					cont_delay(3) += 1.0;
				}
				else {
					Q_delay(3) = Q5d_HP_AV(0);
					for (int i = 0; i < cont_delay(3) - 1; i++) {
						Q5d_HP_AV(i) = Q5d_HP_AV(i + 1);
					}
					Q5d_HP_AV(cont_delay(3) - 1) = Q(2);
				}
			}
			
			if (delay_size(4) < 1.0) {
				Q_delay(4) = Q(0);
			}
			else {
				if (t < delay(4)) {
					Q_delay(4) = Q_old(0) - delay(4) * (Q(0) - Q_old(0)) / sst;
					Q1d_SA_HP(cont_delay(4)) = Q_delay(4);
					cont_delay(4) += 1.0;
				}
				else {
					Q_delay(4) = Q1d_SA_HP(0);
					for (int i = 0; i < cont_delay(4) - 1; i++) {
						Q1d_SA_HP(i) = Q1d_SA_HP(i + 1);
					}
					Q1d_SA_HP(cont_delay(4) - 1) = Q(0);
				}
			}
			
			if (delay_size(5) < 1.0) {
				Q_delay(5) = Q(1);
			}
			else {
				if (t < delay(5)) {
					Q_delay(5) = Q_old(1) - delay(5) * (Q(1) - Q_old(1)) / sst;
					Q3d_AV_HP(cont_delay(5)) = Q_delay(5);
					cont_delay(5) += 1.0;
				}
				else {
					Q_delay(5) = Q3d_AV_HP(0);
					for (int i = 0; i < cont_delay(5) - 1; i++) {
						Q3d_AV_HP(i) = Q3d_AV_HP(i + 1);
					}
					Q3d_AV_HP(cont_delay(5) - 1) = Q(1);
				}
			}
			Q_old = { Q(0), Q(1), Q(2) };
            rk4(t, t + sst, Q, Qp, Q_delay, u);
            t = t + sst;
        }

		edo(t, cst, Q, Qp, Q_delay, Qpp, u);
		ECG = 1.0 + 0.06 * Q(0) + 0.1 * Q(1) + 0.3 * Q(2);
		ECGp = 0.06 * Qp(0) + 0.1 * Qp(1) + 0.3 * Qp(2);
		ECGpp = 0.06 * Qpp(0) + 0.1 * Qpp(1) + 0.3 * Qpp(2);

        if(t>=atf && t * 0.1048 >= 14.0 && t * 0.1048 < 20.0){
			fprintf(saida1, "%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t", t * 0.1048, ECG, ECGp, u, xd, xpd, e, ep, s, d_est, norm(weight, 2));
			fprintf(saida1, "%.6f\t%.6f\t%.6f\t", Q(0), Q(1), Q(2));
			fprintf(saida1, "%.6f\t%.6f\t%.6f\t", Qp(0), Qp(1), Qp(2));
			fprintf(saida1, "\n"); 
			fprintf(saidaw, "%.6f\t", t);
			for (int i = 0; i < n; i++) {
				fprintf(saidaw, "%.6f\t", weight(i));
			}
			fprintf(saidaw, "\n");
        }

	}

	fclose(saida1);
	fclose(saidaw);
	file_in.close();

	return 0;
}

double sinal(double x) {
	if (fabs(x) > 0.0) {
		return x / fabs(x);
	}
	else {
		return 0.0;
	}
}
