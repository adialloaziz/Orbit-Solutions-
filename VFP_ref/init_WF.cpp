//shoudl return:  free_space_and_parallel_plates_integ_part_fftfactor (complex<double> *)
//write in file "impedance.txt"

#include<iostream>
#include<fstream>
#include<math.h>
#include<complex>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_integration.h>
#include <fftw3.h>

using namespace std;

double v(double q, double du_sur_dq);
double G2(double q, double dx_sur_dq, int k_max);
double root(double x, double k);
double w(double q, double du_sur_dq);

//to use gsl
double v_wrapper(double q,void* parptr); //before name: v_Cfunction
struct Params {
    double du_sur_dq;
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char * argv[])
{
  //_______________electron parameters_____________________
  double R=5.36; //m
  double h=1.25e-2; //m (half the height between the two parallel plates)
  double fs=1467; //sync. fq. s-1
  double omega_s=2.*M_PI*1467;
  double T0=1.181e-6; //s
  double E0=2.75e9; //eV
  double energy_spread=1.017e-3; //normalized by E0
  double sigma_e = energy_spread * E0; //eV
  double sigma_z = 1.45e-3; //m
  double _e=1.602e-19; //C
  double gamma = E0/(_e * 0.511e6);
  
  //_____________mesh & wakefield par.  (n=grand_n; L=grand_L) ____________
  double Lq=20; //unit of sigma_z
  int    nq=480;
  double grand_L=40;
  int grand_n = nq*grand_L/Lq;
  int super_n=100*grand_n;
  int n_sur_n = super_n/grand_n;
  double dq = grand_L/double(grand_n); // pas de la grille
    
  int k_max=150;
  
  //___________ declaration _______________________________________________
  double* force = new double[nq];
  double* wake_potential = new double[nq];
  complex<double>* impedance_final=new complex<double>[grand_n/2+1];
  double* wf_free_space_and_parallel_plate_integ_part = new double[grand_n];
   
  //_________ parametres pour l'expression du wakefield ______________________
  const double epsilon_o = 8.854e-12;
  double Delta = h/R; // h : demi-hauteur de la chambre, R : rayon de courbure des aimants
  double du_sur_dq = 3.*pow(gamma,3.)*sigma_z/(2.*R); // chgt de variable pour la partie free space
  double dx_sur_dq = sigma_z/(2.*R*pow(Delta,1.5)); // chgt de variable pour la partie parallel plate
  double cte_free_space = (T0/(4.*M_PI*epsilon_o))*(4./3.)*pow(gamma,4.)/(R*R);
  double cte_parallel_plate = -(T0/(8.*M_PI*epsilon_o*h*h));

  //________Fee space wakefield strength at the value 0 _______________
  double _v = v(dq/2., du_sur_dq);

  //____________ Tableaux des wakefunctions (calculés une seule fois !)  __________________
  double* tab_w_without_zero = new double[super_n];
  double* tab_G2 = new double[super_n];
  // integration by part
  double* _k=new double[super_n];
  double* tab_v=new double[super_n];
  complex<double>* tf_v=new complex<double>[super_n];
    
  double super_dq=dq/double(n_sur_n);

  for (int i=0; i<super_n; i++)
    {
      double q;
      if (i<super_n/2+1)
        {
	  q = double(i)*super_dq;
        }else
        {
	  q = -grand_L + double(i)*super_dq;
        }

      tab_G2[i] = G2(q, dx_sur_dq, k_max);
      tab_v[i]=v(q, du_sur_dq); 

	if (q<dq/2.)
        {
            tab_w_without_zero[i] = 0;
        }else
        {
	  tab_w_without_zero[i] = w(q, du_sur_dq);
        }
    }
  
  
  // calcul de la moyenne de la fonction v entre -super_dq/2 et super_dq/2
  // en pratique: 0.5*[integral(v(x),0..dq/2)/super_dq], car v(q)<0 pour q<0
    
  Params p;
  p.du_sur_dq =du_sur_dq;
  gsl_function F;
  F.function = &v_wrapper;
  F.params = &p;
  
  double result, error;
  size_t neval;
  const double epsabs=1e-4;
  const double epsrel=1e-4;
  int ierr=gsl_integration_qng (&F,0.,0.5*super_dq,epsabs,epsrel,&tab_v[0],&error,&neval);
  if(ierr){cout<<"gsl_integration_qng returned error "<<ierr<<endl;}
  tab_v[0]=0.5*tab_v[0]/(0.5*super_dq);// 0.5*moyenne entre 0 et super_dq/2  
  

  complex<double>* tf_w_without_zero = new complex<double>[super_n/2+1];
  fftw_complex *out_without_zero     = (fftw_complex*)tf_w_without_zero;
  complex<double> *tf_G2             = new complex<double>[super_n/2+1];
  fftw_complex *out_G2               = (fftw_complex*)tf_G2;
  complex<double> *tmp_tf            = new complex<double>[grand_n/2+1];
  fftw_complex * out                 = (fftw_complex*)tmp_tf;
  double* tab_n                      = new double[grand_n];
 
  complex<double> *free_space_and_parallel_plates_integ_part_fftfactor=new complex<double>[grand_n/2+1];
    
  fftw_plan pfor_super_n= fftw_plan_dft_r2c_1d(super_n,tab_w_without_zero,out_without_zero,FFTW_ESTIMATE);
  fftw_execute_dft_r2c(pfor_super_n,tab_w_without_zero,out_without_zero);
  fftw_execute_dft_r2c(pfor_super_n,tab_G2,out_G2);
  fftw_execute_dft_r2c(pfor_super_n,tab_v,(fftw_complex*)tf_v);
  
  fftw_plan pfor  = fftw_plan_dft_r2c_1d(grand_n,tab_n,out,FFTW_ESTIMATE);
  fftw_plan pback = fftw_plan_dft_c2r_1d(grand_n,out,tab_n,FFTW_ESTIMATE);
  
  //before:  Init_k();
  for (int iy=0; iy<super_n; iy++)
    {
      _k[iy] = double(iy)*2*M_PI/grand_L;
    }
  
  //before in   compute_free_space_and_parallel_plates_integ_part_fftfactor();
  for (int i=0; i<grand_n/2+1; i++){
    free_space_and_parallel_plates_integ_part_fftfactor[i]= dq/double(super_n)*complex<double>(0.,_k[i])*tf_v[i]*cte_free_space/du_sur_dq;
    free_space_and_parallel_plates_integ_part_fftfactor[i]+=dq/double(super_n)*tf_G2[i]*cte_parallel_plate;
  }

  //write result
  ofstream o1; o1.open("impedance.txt");
  o1.precision(15);
  for(int i=0; i<grand_n/2 +1; i++) {o1<<real(free_space_and_parallel_plates_integ_part_fftfactor[i])<<" "<<imag(free_space_and_parallel_plates_integ_part_fftfactor[i])<<endl;}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double v(double q, double du_sur_dq)
{
  double u = du_sur_dq*q;
  double lambda = sqrt(u*u+1);
  double res=0;
  if(q>0){
    res = 9./16.*(-2./u + 1./(u*lambda)*(pow(lambda+u,1./3)+pow(lambda+u,-1./3)) + 2./lambda*(pow(lambda+u,2./3)-pow(lambda+u,-2./3)));
  }
  return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double G2(double q, double dx_sur_dq, int k_max)
{
  double x = dx_sur_dq*q;
  double Y;
  double k;
  double res = 0;
  for (int ik=1; ik<=k_max; ik++)
    {
      k = double(ik);
      Y = root(x,k);
      res = res + 2.*pow(-1,k+1)/(k*k)*(4.*pow(Y,4.)*(3.-pow(Y,4.))/pow(1.+pow(Y,4.),3.));
    }
  
  return res;
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Roots of Y^4 - 6*Y*x/k^(3/2) - 3 = 0
double root(double x, double k)
{
  double res;
  
    int n = 5;
    //Coefficients of P(x)= Y^4 - 6*Y*x/k^(3/2) - 3
    double * a = new double[n];
    a[0] = -3;
    a[1] = - 6.*x/pow(k,3./2);
    a[2] = a[3] = 0;
    a[4] = 1.;
    
    gsl_poly_complex_workspace * W = gsl_poly_complex_workspace_alloc(n);
    double * z = new double[2*(n-1)];
    gsl_poly_complex_solve(a,n,W,z);
    
    double * tmp = new double[2];
    if ((z[1]==0) && (z[3]==0))
    {
        tmp[0] = z[0];
        tmp[1] = z[2];
    }else if ((z[1]==0) && (z[5]==0))
    {
        tmp[0] = z[0];
        tmp[1] = z[4];
    }else if ((z[1]==0) && (z[7]==0))
    {
        tmp[0] = z[0];
        tmp[1] = z[6];
    }else if ((z[3]==0) && (z[5]==0))
    {
        tmp[0] = z[2];
        tmp[1] = z[4];
    }else if ((z[3]==0) && (z[7]==0))
    {
        tmp[0] = z[2];
        tmp[1] = z[6];
    }else if ((z[5]==0) && (z[7]==0))
    {
        tmp[0] = z[4];
        tmp[1] = z[6];
    }
    
    if (x<0)
    {
        res = fmin(abs(tmp[0]),abs(tmp[1]));
    }else
    {
        res = fmax(abs(tmp[0]),abs(tmp[1]));
    }
    
    gsl_poly_complex_workspace_free(W);
    delete[] tmp;
    delete[] a;
    delete[] z;

    return res;
}


///////////////////////////////////////////////////////////////


double w(double q, double du_sur_dq)
{
    double u = du_sur_dq*q;
    double lambda = sqrt(u*u+1);
    double res;
    if (q<0)
    {
        res = 0;
    }else if (q==0)
    {
        res = 0.5;
    }else
    {
        res = 9./(8.*u*u) - 9/(16*u*u*lambda)*(pow(lambda+u,1./3)+pow(lambda+u,-1./3)) - 9/(16*lambda*lambda*lambda)*(pow(lambda+u,1./3)+pow(lambda+u,-1./3)) + 3/(16*u*lambda)*(u/lambda+1)*(pow(lambda+u,-2./3)-pow(lambda+u,-4./3)) - 9*u/(8*lambda*lambda*lambda)*(pow(lambda+u,2./3)-pow(lambda+u,-2./3)) + 3/(4*lambda)*(u/lambda+1)*(pow(lambda+u,-1./3)+pow(lambda+u,-5./3));
    }
    return res;
}


////////////////////////////////////////////////////////////////////
// version pouvant ete appellée par la gsl.
double v_wrapper(double q, void *params){
  Params *p = static_cast<Params*>(params);
  return v(q, p->du_sur_dq);
}

///////////////////////////////////////////////////////////
