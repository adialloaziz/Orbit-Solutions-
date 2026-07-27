#include "../include/vfp.h"


using namespace std;

//need to have for argument:
//1) tab "free_space_and_parallel_plates_integ_part_fftfactor" (calculated by init_WF.cpp)
//2) initial distribution U[nq*np]

void integ_vfp(double t1, double t2){
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
  // double t1=0;
  // double t2=63.118; //norm. time (?)
  
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


// int main(int argc, char * argv[])
// {

//   //output
//   ofstream o_info; o_info.open("info.txt");
//   ofstream o_tmp; o_tmp.open("tmp.txt");
  
  
//   //----------parameters---------------
//   //___ spatial mesh___
//   double Lq=20; //unit of sigma_z
//   double Lp=20; //unit of sigma_E
//   double grand_L=40;
//   int    nq= 480;
//   int    np= 480;
  
//   double dq = Lq/double(nq);
//   double dp = Lp/double(np);
    
//   double* q = new double[nq];
//   double* p = new double [np];
//   for (int i=0; i<nq; i++){q[i]=-Lq/2. + double(i)*dq;}
//   for (int i=0; i<np; i++){p[i]=-Lp/2. + double(i)*dp;}
//   int grand_n = nq*grand_L/Lq;
  
//   int neqn = nq*np;
  
//   //___temporal par. ___
//   int nt=8000;
//   int ntint=10;
//   double t1=0;
//   double t2=63.118; //norm. time (?)
  
//   int ntot = nt*ntint;
//   double dt = (t2-t1)/double(ntot); 
//   double t;

//   //___ electron par. ___
//   double _e=1.602e-19; //C
//   double E0=2.75e9; //eV
//   double energy_spread=1.017e-3; //normalized by E0
//   double I=0.44e-3; //A (ou 0.44 ?)
//   double fs=1467; //sync. fq. s-1
//   double omega_s=2.*M_PI*fs; 
//   double td=3.27e-3; //s
//   double T0=1.181e-6; //s
    
//   double sigma_e = energy_spread * E0*_e; //C
//   double sigma_z = 1.45e-3; //unit ?
//   double gamma = E0/0.511e6; cout<<"gamma="<<gamma<<endl;
//   double epsilon = 1./(omega_s*td);
//   double Nelectrons=I*T0/_e;
//   cout<<"Number of electrons per bunch: "<<Nelectrons<<endl;
//   double* density = new double[nq];

//   //___ wakefield___
//   //before : Wake_1 = new Wakefield(grand_n,grand_L,R,gamma,sigma_z,h,To,k_max,inductance,resistance);
//   complex<double> *free_space_and_parallel_plates_integ_part_fftfactor=new complex<double>[grand_n/2+1]; //to replace by input argument
//   //read instead
//   ifstream i1; i1.open("impedance.txt"); i1.precision(15);
//   double re, im;
//   for(int iq=0; iq<grand_n/2 +1; iq++) {
//     i1>>re>>im;
//     free_space_and_parallel_plates_integ_part_fftfactor[iq]= {re, im};
//   }
  
//   double R=5.36; //m
//   double Ic1 =_e*2.*M_PI*R/(omega_s*sigma_e*T0);
//   //          c1 =_e*2*M_PI*R/(omega_s*sigma_e*To); cout<<"Ic1="<<Ic1<<endl;

//   //for wakefield calculation (mostly declaration)
//   double* tab_n                      = new double[grand_n];
//   complex<double> *tmp_tf            = new complex<double>[grand_n/2+1];
//   fftw_complex * out                 = (fftw_complex*)tmp_tf;

//   fftw_plan pfor  = fftw_plan_dft_r2c_1d(grand_n,tab_n,out,FFTW_ESTIMATE);
//   fftw_plan pback = fftw_plan_dft_c2r_1d(grand_n,out,tab_n,FFTW_ESTIMATE);
  
//   complex<double>* impedance_final=new complex<double>[grand_n/2+1];
//   complex<double> * tf_g             = new complex<double>[grand_n/2+1];
//   double * wf_free_space_and_parallel_plate_integ_part = new double[grand_n];
//   double*  force = new double[nq];

//   cout<<"Ic1="<<Ic1<<" sigma_e="<<sigma_e<<" omega_s="<<omega_s<<" T0="<<T0<<endl;
//   for(int i=0; i<grand_n/2+1; i++){impedance_final[i]=Ic1*free_space_and_parallel_plates_integ_part_fftfactor[i]; o_tmp<<real(impedance_final[i])<<" "<<imag(impedance_final[i])<<endl;}
  
//   //___ info ___
//   complex<double> * tf_density  = new complex<double>[nq];
//   fftw_complex *    out_tf_density = (fftw_complex*)tf_density;
//   fftw_plan pfor_info = fftw_plan_dft_r2c_1d(nq ,density, out_tf_density,FFTW_ESTIMATE);

//   double    cutoff=2.5;
//   double    ko        = cutoff;        // cm-1
//   double    k         = sigma_z*(100.)*ko; // *100 -> sigma_z en cm
//   int       k_bolo    = int(k*Lq);
    
//   //____initial condition: gauss - to do: from input argument____
//   double * U0=new double[nq*np]; //;  double * U_tmp;
//   double * U1=new double[nq*np]; //;  double * U_tmp;

//   //before: init_table_ci(U, U_tmp,(void*)& par);
//   // for (int i=0; i<np*nq; i++)U[i]=0;
//   double rms_gauss_q = 1.2;
//   double rms_gauss_p = 1.2;

//   for (int iq=0; iq<nq; iq++)
//     {
//       double q1 =q[iq];
//       for (int ip=0; ip<np; ip++)
// 	{
// 	  double p1 = p[ip];
// 	  U0[iq*np+ip] = (1./(2.*rms_gauss_q*rms_gauss_p*M_PI))*exp(-(q1*q1)/(2.*rms_gauss_q*rms_gauss_q))*exp(-(p1*p1)/(2.*rms_gauss_p*rms_gauss_p));
// 	}
//     }

//   //projection to get initial density
//   //par.Projection_plusshotnoise_MPItorootonly(U,par.density,par.density_addednoise,0);
//   //  double* density=new double[nq];
//   for (int iq=0; iq<nq; iq++)
//     {
//       density[iq]=0;
//       for (int ip=0; ip<np-1; ip++) {density[iq] += (dp/2.)*(U0[iq*np+ip]+U0[iq*np+ip+1]);}
//     }
  
//   //_________________temporal integration _________________________
//   for (int i=0; i<nt; i++)
//     {
//       double tf = t1 + double((i+1)*ntint)*dt;
//       cout<<"iteration n° : "<<i+1<<" / "<<nt<<endl;
      
//       for (int j=0; j<ntint; j++)
// 	{
// 	  t=t1+ double(i*ntint+j)*dt;
// 	  //before: integration_finite_diff_MPI(U,U_tmp,(void*)&par, t1+ double(i*ntint+j)*par.dt);  //void integration_finite_diff_MPI(double * U0, double * U1, void * parptr, double t)
	  
// 	  // Calcul de la force collective due au CSR (result in force) - // Attention, la densité doit avoir été calculée au préalable (résultat dans par->densite). Voir fin de la fonction.
// 	  force_co_from_projection_plus_shot_noise(U0, dt, t, grand_n, nq,  np, density, wf_free_space_and_parallel_plate_integ_part,  impedance_final,  pfor, pback,  tf_g,  tmp_tf, I, force );
// 	  //for(int iq=0; iq<nq; iq++){force[iq]=0;}
	  	  
// 	  // Calcul du terme de Vlasov. U0 -> U1, U0 est inchangé
// 	  vlasov(dt, U0, U1, nq,  np,  dq, dp,  q,  p, force);
	    
// 	  // Calcul du terme de Fokker-Planck. 
// 	  // Attention: U1 calculé à partir de U0 et U1: partir de U1=U1+FP(U0)
// 	  fokker_planck_finite_diff_MPI(dt, U0, U1, dp, nq, np,  epsilon,  p);
	  	  
// 	  //U1-> U0
// 	  for (int iq =0; iq<nq; iq++)
// 	    {
// 	      for (int ip=0; ip<np; ip++)
// 		{
// 		  U0[iq*np+ip]=U1[iq*np+ip];
// 		}
// 	    }	  
	  
// 	  //projection
// 	  for (int iq=0; iq<nq; iq++)
// 	    {
// 	      density[iq]=0;
// 	      for (int ip=0; ip<np-1; ip++) {density[iq] += (dp/2.)*(U0[iq*np+ip]+U0[iq*np+ip+1]);}
// 	    }	  
// 	}//j
      
      
//       //__info__
//       //thz
//       fftw_execute(pfor_info);
//       double Pthz = 0.;
//       for (int i=k_bolo; i<nq/2; i++){Pthz += (norm(tf_density[i]) + norm(tf_density[i+1]))/2.; }
  
//       //
//       o_info<<i<<" "<<tf<<" "<<info_area(nq, density, dq)<<" "<<info_rms(nq, dq, density, q)<<" "<<Pthz<<endl;
      
//     }//i
// }
