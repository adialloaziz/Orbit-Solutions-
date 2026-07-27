#include "../include/utility.h"



void force_co_from_projection_plus_shot_noise(double * U, double dt, double t, int grand_n, int nq, int np, double* density, double*  wf_free_space_and_parallel_plate_integ_part,  complex<double>* impedance_final, fftw_plan pfor, fftw_plan pback,  complex<double> *  tf_g,  complex<double> *  tmp_tf, double I, double* force )
{
  double * tmp     = new double[grand_n];
  
  //density in grand_n dimension
  for(int i=0; i<grand_n; i++)
    {
      if( (i<(grand_n-nq)/2) || (i>=nq+(grand_n-nq)/2) ) { tmp[i] = 0.;}
      else
	{
	  tmp[i] = density[i-(grand_n-nq)/2];
	}
    }
  
  //Bending magnet  (covonlution between rho(=tmp) and impedance)
  //void free_space_and_parallel_plates_integ_part(double * g, double * convol, complex<double>* impedance, fftw_plan pfor, )
  free_space_and_parallel_plates_integ_part(tmp, wf_free_space_and_parallel_plate_integ_part, impedance_final, pfor, pback, grand_n, tf_g, tmp_tf); //result in  wf_free_space...
  
  
  for (int iq=0; iq<nq; iq++)
    {
      force[iq]= I*wf_free_space_and_parallel_plate_integ_part[iq+(grand_n-nq)/2];
      //par->wake_potential[iq]=tmp[iq]; //for FB
    }
  
  delete[] tmp;

}

///////////////////////////////////////////////////////////////////////

//  free_space_and_parallel_plates_integ_part(tmp, par->wf_free_space_and_parallel_plate_integ_part, par->impedance_final);

//free_space_and_parallel_plates_integ_part(tmp, wf_free_space_and_parallel_plate_integ_part, impedance_final, pfor, pback, nq, tf_g, tmp_tf); //result in  wf_free_space...

void free_space_and_parallel_plates_integ_part(double * g,  double * convol, complex<double>* impedance, fftw_plan pfor, fftw_plan pback, int grand_n,     complex<double> * tf_g,  complex<double> *tmp_tf )
{
  
  fftw_complex * out_g  = (fftw_complex*)tf_g;
  fftw_complex * out    = (fftw_complex*)tmp_tf;
  
  fftw_execute_dft_r2c(pfor, g, out_g); //tf of tmp (density in grand_n)
  
  for (int i=0; i<grand_n/2+1; i++)
    {
      tmp_tf[i] = impedance[i]*tf_g[i];
    }
  
  fftw_execute_dft_c2r(pback,out,convol);
  
}

////////////////////////////////////////////////////////
// vlasov term
void vlasov(double dt, double * U0, double * U1, int nq, int np, double dq, double dp, double* q, double* p, double* force)
{
  for (int iq=0; iq<nq; iq++)
    {
      int k, l;
      double x, y;
      double Zij[2];
      double MZij[2];
      Zij[0] = q[iq];

      for (int ip=0; ip<np;ip++)
        {
	  Zij[1] = p[ip];
          
	  MZij[0] = Zij[0]*cos(dt)    + (Zij[1]+ force[iq]*dt )*sin(dt); 
	  MZij[1] = -(Zij[0])*sin(dt) + (Zij[1]+ force[iq]*dt )*cos(dt);	  
	  
      
	  // integer part
	  k = int((MZij[0]- q[0])/dq);
	  l = int((MZij[1]- p[0])/dp);
      
	  if((k-1)>=0 && (k+1)<nq && (l-1)>=0 && (l+1)<np )
	    {
	      x = (MZij[0]- q[k])/dq;
	      y = (MZij[1]- p[l])/dp;
	      int i11=(k-1)*np+(l-1);
	      int i12=(k-1)*np+l;
	      int i13=(k-1)*np+(l+1);
	      int i21=k*np+(l-1);
	      int i22=k*np+l;
	      int i23=k*np+(l+1);
	      int i31=(k+1)*np+(l-1);
	      int i32=(k+1)*np+l;
	      int i33=(k+1)*np+(l+1);
	  
	  
	      U1[iq*np+ip] = (1./4.)*(
				      x*(x-1.) * ( y*(y-1.)*U0[i11] + 2.*(1.-y*y)*U0[i12] + y*(y+1.)*U0[i13] ) +
				      2.*(1.-x*x) * ( y*(y-1.)*U0[i21] + 2.*(1.-y*y)*U0[i22] + y*(y+1.)*U0[i23] ) +
				      x*(x+1.) * ( y*(y-1.)*U0[i31] + 2.*(1.-y*y)*U0[i32] + y*(y+1.)*U0[i33] ) );
	    }
	  else{U1[iq*np +ip]=0;}
	}
    }
}

//////////////////////////////////////////////////////
void fokker_planck_finite_diff_MPI(double dt, double * U0, double * U1, double dp, int nq, int np, double epsilon, double* p)
{
  for (int iq=0; iq<nq; iq++)
    {
      double Gij1, Gij;
      for (int ip=1; ip<np-1; ip++)
        {
	  Gij1 = (1./dp)*(U0[iq*np+(ip+1)]-U0[iq*np+ip]) + (p[ip+1]/2.)*(U0[iq*np+(ip+1)]+U0[iq*np+ip]);
	  Gij = (1./dp)*(U0[iq*np+ip]-U0[iq*np+(ip-1)]) + (p[ip]/2.)*(U0[iq*np+ip]+U0[iq*np+(ip-1)]);
	  double fpterm = (2.*epsilon/dp)*(Gij1 - Gij);
	  U1[iq*np+ip] = U1[iq*np+ip] + dt*fpterm;
        }
    }
}

////////////////////////////////////////////////////////
double info_area(int nq, double* f, double dq)
{
  double res = 0.;
  for (int i=0; i<nq-1; i++)
    {
      res += dq/2. * (f[i] + f[i+1]);
    }
  return res;
}


///////////////////////////////////
double info_rms(int nq, double dq, double* f, double* q)
{
  
  double mean = 0.;
  for (int i=0; i<nq-1; i++)
    {
      mean += dq/2. * (q[i]*f[i] + q[i+1]*f[i+1]);
    }
  
  double res = 0.;
  
  for (int i=0; i<nq-1; i++)
    {
      res += dq/2. * (q[i]*q[i]*f[i] + q[i+1]*q[i+1]*f[i+1]);
    }
  
  res = sqrt(res-mean*mean);
  return res;
}

//-------Functions used in the init_WF code-----------------------------
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
