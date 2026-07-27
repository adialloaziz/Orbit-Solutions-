#ifndef UTILITY_H
#define UTILITY_H

#pragma once
#include <complex>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_integration.h>
#include <fftw3.h>
using namespace std;

struct Params {
    double du_sur_dq;
};

//-------------------------Functions used in the VFP code-----------------------------
void free_space_and_parallel_plates_integ_part(double * g,  double * convol, complex<double>* impedance, fftw_plan pfor, fftw_plan pback, int nq,     complex<double> * tf_g,  complex<double> *tmp_tf);
//1) tab "free_space_and_parallel_plates_integ_part_fftfactor" (calculated by init_WF.cpp)

void force_co_from_projection_plus_shot_noise(double * U, double dt, double t, int grand_n, int nq, int np, double* density, double*  wf_free_space_and_parallel_plate_integ_part,  complex<double>* impedance_final, fftw_plan pfor, fftw_plan pback,  complex<double> *  tf_g,  complex<double> *  tmp_tf, double I, double* force );
void vlasov(double dt, double * U0, double * U1, int nq, int np, double dq, double dp, double* q, double* p, double* force);
void fokker_planck_finite_diff_MPI(double dt, double * U0, double * U1, double dp, int nq, int np, double epsilon, double* p);
double info_area(int nq, double* f, double dq);
double info_rms(int nq, double dq, double* f, double* q);

//-------------------------Functions used in the init_WF code-----------------------------
double v(double q, double du_sur_dq);
double G2(double q, double dx_sur_dq, int k_max);
double root(double x, double k);
double w(double q, double du_sur_dq);
//to use gsl
double v_wrapper(double q,void* parptr); //before name: v_Cfunction

#endif //UTILITY_H