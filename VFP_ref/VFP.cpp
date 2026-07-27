#include<iostream>
#include<fstream>
#include<math.h>
#include<complex>
#include <fftw3.h>

using namespace std;

//need to have for argument:
//1) tab "free_space_and_parallel_plates_integ_part_fftfactor" (calculated by init_WF.cpp)
//2) initial distribution U[nq*np]

void free_space_and_parallel_plates_integ_part(double * g,  double * convol, complex<double>* impedance, fftw_plan pfor, fftw_plan pback, int nq,     complex<double> * tf_g,  complex<double> *tmp_tf);
void force_co_from_projection_plus_shot_noise(double * U, double dt, double t, int grand_n, int nq, int np, double* density, double*  wf_free_space_and_parallel_plate_integ_part,  complex<double>* impedance_final, fftw_plan pfor, fftw_plan pback,  complex<double> *  tf_g,  complex<double> *  tmp_tf, double I, double* force );
void vlasov(double dt, double * U0, double * U1, int nq, int np, double dq, double dp, double* q, double* p, double* force);
void fokker_planck_finite_diff_MPI(double dt, double * U0, double * U1, double dp, int nq, int np, double epsilon, double* p);
double info_area(int nq, double* f, double dq);
double info_rms(int nq, double dq, double* f, double* q);


int main(int argc, char * argv[])
{

  //output
  ofstream o_info; o_info.open("info.txt");
  ofstream o_tmp; o_tmp.open("tmp.txt");
  
  
  //----------parameters---------------
  //___ spatial mesh___
  double Lq=20; //unit of sigma_z
  double Lp=20; //unit of sigma_E
  double grand_L=40;
  int    nq= 480;
  int    np= 480;
  
  double dq = Lq/double(nq);
  double dp = Lp/double(np);
    
  double* q = new double[nq];
  double* p = new double [np];
  for (int i=0; i<nq; i++){q[i]=-Lq/2. + double(i)*dq;}
  for (int i=0; i<np; i++){p[i]=-Lp/2. + double(i)*dp;}
  int grand_n = nq*grand_L/Lq;
  
  int neqn = nq*np;
  
  //___temporal par. ___
  int nt=8000;
  int ntint=10;
  double t1=0;
  double t2=63.118; //norm. time (?)
  
  int ntot = nt*ntint;
  double dt = (t2-t1)/double(ntot); 
  double t;

  //___ electron par. ___
  double _e=1.602e-19; //C
  double E0=2.75e9; //eV
  double energy_spread=1.017e-3; //normalized by E0
  double I=0.44e-3; //A (ou 0.44 ?)
  double fs=1467; //sync. fq. s-1
  double omega_s=2.*M_PI*fs; 
  double td=3.27e-3; //s
  double T0=1.181e-6; //s
    
  double sigma_e = energy_spread * E0*_e; //C
  double sigma_z = 1.45e-3; //unit ?
  double gamma = E0/0.511e6; cout<<"gamma="<<gamma<<endl;
  double epsilon = 1./(omega_s*td);
  double Nelectrons=I*T0/_e;
  cout<<"Number of electrons per bunch: "<<Nelectrons<<endl;
  double* density = new double[nq];

  //___ wakefield___
  //before : Wake_1 = new Wakefield(grand_n,grand_L,R,gamma,sigma_z,h,To,k_max,inductance,resistance);
  complex<double> *free_space_and_parallel_plates_integ_part_fftfactor=new complex<double>[grand_n/2+1]; //to replace by input argument
  //read instead
  ifstream i1; i1.open("impedance.txt"); i1.precision(15);
  double re, im;
  for(int iq=0; iq<grand_n/2 +1; iq++) {
    i1>>re>>im;
    free_space_and_parallel_plates_integ_part_fftfactor[iq]= {re, im};
  }
  
  double R=5.36; //m
  double Ic1 =_e*2.*M_PI*R/(omega_s*sigma_e*T0);
  //          c1 =_e*2*M_PI*R/(omega_s*sigma_e*To); cout<<"Ic1="<<Ic1<<endl;

  //for wakefield calculation (mostly declaration)
  double* tab_n                      = new double[grand_n];
  complex<double> *tmp_tf            = new complex<double>[grand_n/2+1];
  fftw_complex * out                 = (fftw_complex*)tmp_tf;

  fftw_plan pfor  = fftw_plan_dft_r2c_1d(grand_n,tab_n,out,FFTW_ESTIMATE);
  fftw_plan pback = fftw_plan_dft_c2r_1d(grand_n,out,tab_n,FFTW_ESTIMATE);
  
  complex<double>* impedance_final=new complex<double>[grand_n/2+1];
  complex<double> * tf_g             = new complex<double>[grand_n/2+1];
  double * wf_free_space_and_parallel_plate_integ_part = new double[grand_n];
  double*  force = new double[nq];

  cout<<"Ic1="<<Ic1<<" sigma_e="<<sigma_e<<" omega_s="<<omega_s<<" T0="<<T0<<endl;
  for(int i=0; i<grand_n/2+1; i++){impedance_final[i]=Ic1*free_space_and_parallel_plates_integ_part_fftfactor[i]; o_tmp<<real(impedance_final[i])<<" "<<imag(impedance_final[i])<<endl;}
  
  //___ info ___
  complex<double> * tf_density  = new complex<double>[nq];
  fftw_complex *    out_tf_density = (fftw_complex*)tf_density;
  fftw_plan pfor_info = fftw_plan_dft_r2c_1d(nq ,density, out_tf_density,FFTW_ESTIMATE);

  double    cutoff=2.5;
  double    ko        = cutoff;        // cm-1
  double    k         = sigma_z*(100.)*ko; // *100 -> sigma_z en cm
  int       k_bolo    = int(k*Lq);
    
  //____initial condition: gauss - to do: from input argument____
  double * U0=new double[nq*np]; //;  double * U_tmp;
  double * U1=new double[nq*np]; //;  double * U_tmp;

  //before: init_table_ci(U, U_tmp,(void*)& par);
  // for (int i=0; i<np*nq; i++)U[i]=0;
  double rms_gauss_q = 1.2;
  double rms_gauss_p = 1.2;

  for (int iq=0; iq<nq; iq++)
    {
      double q1 =q[iq];
      for (int ip=0; ip<np; ip++)
	{
	  double p1 = p[ip];
	  U0[iq*np+ip] = (1./(2.*rms_gauss_q*rms_gauss_p*M_PI))*exp(-(q1*q1)/(2.*rms_gauss_q*rms_gauss_q))*exp(-(p1*p1)/(2.*rms_gauss_p*rms_gauss_p));
	}
    }

  //projection to get initial density
  //par.Projection_plusshotnoise_MPItorootonly(U,par.density,par.density_addednoise,0);
  //  double* density=new double[nq];
  for (int iq=0; iq<nq; iq++)
    {
      density[iq]=0;
      for (int ip=0; ip<np-1; ip++) {density[iq] += (dp/2.)*(U0[iq*np+ip]+U0[iq*np+ip+1]);}
    }
  
  //_________________temporal integration _________________________
  for (int i=0; i<nt; i++)
    {
      double tf = t1 + double((i+1)*ntint)*dt;
      cout<<"iteration n° : "<<i+1<<" / "<<nt<<endl;
      
      for (int j=0; j<ntint; j++)
	{
	  t=t1+ double(i*ntint+j)*dt;
	  //before: integration_finite_diff_MPI(U,U_tmp,(void*)&par, t1+ double(i*ntint+j)*par.dt);  //void integration_finite_diff_MPI(double * U0, double * U1, void * parptr, double t)
	  
	  // Calcul de la force collective due au CSR (result in force) - // Attention, la densité doit avoir été calculée au préalable (résultat dans par->densite). Voir fin de la fonction.
	  force_co_from_projection_plus_shot_noise(U0, dt, t, grand_n, nq,  np, density, wf_free_space_and_parallel_plate_integ_part,  impedance_final,  pfor, pback,  tf_g,  tmp_tf, I, force );
	  //for(int iq=0; iq<nq; iq++){force[iq]=0;}
	  	  
	  // Calcul du terme de Vlasov. U0 -> U1, U0 est inchangé
	  vlasov(dt, U0, U1, nq,  np,  dq, dp,  q,  p, force);
	    
	  // Calcul du terme de Fokker-Planck. 
	  // Attention: U1 calculé à partir de U0 et U1: partir de U1=U1+FP(U0)
	  fokker_planck_finite_diff_MPI(dt, U0, U1, dp, nq, np,  epsilon,  p);
	  	  
	  //U1-> U0
	  for (int iq =0; iq<nq; iq++)
	    {
	      for (int ip=0; ip<np; ip++)
		{
		  U0[iq*np+ip]=U1[iq*np+ip];
		}
	    }	  
	  
	  //projection
	  for (int iq=0; iq<nq; iq++)
	    {
	      density[iq]=0;
	      for (int ip=0; ip<np-1; ip++) {density[iq] += (dp/2.)*(U0[iq*np+ip]+U0[iq*np+ip+1]);}
	    }	  
	}//j
      
      
      //__info__
      //thz
      fftw_execute(pfor_info);
      double Pthz = 0.;
      for (int i=k_bolo; i<nq/2; i++){Pthz += (norm(tf_density[i]) + norm(tf_density[i+1]))/2.; }
  
      //
      o_info<<i<<" "<<tf<<" "<<info_area(nq, density, dq)<<" "<<info_rms(nq, dq, density, q)<<" "<<Pthz<<endl;
      
    }//i
}

///////////////////////////:
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
////////////////////////////////////////////////////////////////////

