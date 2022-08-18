#define M_PI 3.14159265358979323846

double xt1_SA_AV_old = 0.0;
double xt1_SA_HP_old = 0.0;
double xt3_AV_SA_old = 0.0;
double xt3_AV_HP_old = 0.0;
double xt5_HP_SA_old = 0.0;
double xt5_HP_AV_old = 0.0;

double alp_SA = 3.0;
double nu1_SA = 1.0;
double nu2_SA = -1.9;
double d_SA = 1.9;
double e_SA = 0.55;

double alp_AV = 7.0;
double nu1_AV = 0.5;
double nu2_AV = -0.5;
double d_AV = 4.0;
double e_AV = 0.67;

double alp_HP = 7.0;
double nu1_HP = 1.65;
double nu2_HP = -2.0;
double d_HP = 7.0;
double e_HP = 0.67;

double rho_SA = 8.0;
double rho_HP = 0.0;
double omega_SA = 2.1;
double omega_HP = 0.0;
double k_SA_AV = 0.66;
double k_AV_HP = 14.0;
double kt_SA_AV = 0.09;
double kt_AV_HP = 38.0;
double tau_SA_AV = 0.8;
double tau_AV_HP = 0.1;
double beta_t = 0.0230;

double rho_AV = 0.0;
double omega_AV = 0.0;
double k_AV_SA = 0.0;
double kt_AV_SA = 0.0;
double k_HP_SA = 0.0;
double kt_HP_SA = 0.0;
double k_HP_AV = 0.0;
double kt_HP_AV = 0.0;
double k_SA_HP = 0.0;
double kt_SA_HP = 0.0;
double N_AV_HP = 1.0;
double N_SA_AV = 1.0;
double tau_SA_HP = 0.0;
double tau_AV_SA = 0.0;
double tau_HP_SA = 0.0;
double tau_HP_AV = 0.0;

arma::vec Q1d_SA_HP, Q1d_SA_AV, Q3d_AV_SA, Q3d_AV_HP, Q5d_HP_SA, Q5d_HP_AV;
arma::vec delay(6), delay_size(6), cont_delay(6, fill::zeros);

bool flag_SA_AV = true, flag_AV_HP = true;
double xt1_SA_AV, xt3_AV_HP;

double sgn(double x)
{
	if (x > 0.0)
	{
		return 1.0;
	}else if (x < 0.0)
	{
		return -1.0;
	}else{
		return 0.0;
	}
}

double sat(double x)
{
	if (x > 1.0)
	{
		return 1.0;
	}
	else if (x < -1.0)
	{
		return -1.0;
	}
	else
	{
		return x;
	}

}

double FSA(double t) {
	return rho_SA * sin(omega_SA * t);
}
double FAV(double t) {
	return rho_AV * sin(omega_AV * t);
}
double FHP(double t) {
	return rho_HP * sin(omega_HP * t);
}

void edo(double t, double dt, arma::vec Q, arma::vec Qp, arma::vec Q_delay, arma::vec& Qpp, double u) {
	double x1, x2, x3, x4, x5, x6, x2p, x4p, x6p;

	/*
	0 - SA
	1 - AV
	2 - HP
	*/

	x1 = Q(0);
	x2 = Qp(0);
	x3 = Q(1);
	x4 = Qp(1);
	x5 = Q(2);
	x6 = Qp(2);

	x2p = FSA(t) - alp_SA * x2 * (x1 - nu1_SA) * (x1 - nu2_SA) - (x1 * (x1 + d_SA) * (x1 + e_SA)) / (d_SA * e_SA) - k_AV_SA * x1 + kt_AV_SA * Q_delay(0) - k_HP_SA * x1 + kt_HP_SA * Q_delay(1);
	
	x4p = FAV(t) - alp_AV * x4 * (x3 - nu1_AV) * (x3 - nu2_AV) - (x3 * (x3 + d_AV) * (x3 + e_AV)) / (d_AV * e_AV) - k_SA_AV * x3 + kt_SA_AV * Q_delay(2) - k_HP_AV * x3 + kt_HP_AV * Q_delay(3);

	x6p = FHP(t) - alp_HP * x6 * (x5 - nu1_HP) * (x5 - nu2_HP) - (x5 * (x5 + d_HP) * (x5 + e_HP)) / (d_HP * e_HP) - k_SA_HP * x5 + kt_SA_HP * Q_delay(4) - k_AV_HP * x5 + kt_AV_HP * Q_delay(5) + u;

	Qpp(0) = x2p;
	Qpp(1) = x4p;
	Qpp(2) = x6p;

}

void rk4(double t, double tf, arma::vec& Q, arma::vec& Qp, arma::vec Q_delay, double u){

    arma::vec k1p(3), k2p(3), k3p(3), k4p(3), k1v(3), k2v(3), k3v(3), k4v(3);
    arma::vec Qpp(3), k_delay(6);
	double dt = tf - t;

	/*
	0 - SA
	1 - AV
	2 - HP
	*/

    edo(t, dt, Q, Qp, Q_delay, Qpp, u);
    k1p = dt*Qp;
    k1v = dt*Qpp;
	k_delay = { k1p(1), k1p(2), k1p(0), k1p(2), k1p(0), k1p(1) };

	edo(t + dt / 2.0, dt, Q + k1p / 2.0, Qp + k1v / 2.0, Q_delay + k_delay / 2.0, Qpp, u);
    k2p = dt*(Qp + k1v/2.0);
    k2v = dt*Qpp;
	k_delay = { k2p(1), k2p(2), k2p(0), k2p(2), k2p(0), k2p(1) };

    edo(t + dt/2.0, dt, Q + k2p/2.0, Qp + k2v/2.0, Q_delay + k_delay / 2.0, Qpp, u);
    k3p = dt*(Qp + k2v/2.0);
    k3v = dt*Qpp;
	k_delay = { k3p(1), k3p(2), k3p(0), k3p(2), k3p(0), k3p(1) };

    edo(t + dt, dt, Q + k3p, Qp + k3v, Q_delay + k_delay, Qpp, u);
    k4p = dt*(Qp + k3v);
    k4v = dt*Qpp;

    Q = Q + (k1p + 2.0*k2p + 2.0 * k3p + k4p)/6.0;
    Qp = Qp + (k1v + 2.0*k2v + 2.0 * k3v + k4v)/6.0;

}

void ativ(double x, arma::vec center, arma::vec width, int n, arma::vec& out)
{

	for (int i = 0; i < n; i++) {
		out(i) = exp(-(x - center(i)) * (x - center(i)) / (2.0 * width(i) * width(i)));
	}

}

void atual(double t, double tf, arma::vec& weight, arma::vec center, arma::vec width, double learn, double s, int contador_max, int* cont_ann, arma::vec d, double lim_weight, double lim_s, int training_opt) {

	double h;
	int contador;
	h = tf - t;

	contador = *cont_ann;

	if (training_opt == 0) {//algoritmo de projecao
		if (norm(weight, 2) < lim_weight || (norm(weight, 2) == lim_weight && learn * s * h * dot(weight, d) <= 0.0)) {
			weight = weight + learn * s * h * d;
		}
		else {
			weight = weight + (eye(6, 6) - (weight * weight.t()) / dot(weight, weight)) * learn * s * h * d;
		}
	}
	else if (training_opt == 1) {//algoritmo da zona morta
		if (fabs(s) >= lim_s) {
			weight = weight + learn * s * h * d;
		}
	}
	else {//tradicional
		if (contador <= contador_max) {
			weight = weight + learn * s * h * d;
			contador++;
		}
	}

	*cont_ann = contador;

}