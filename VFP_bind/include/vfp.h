#ifndef VFP_H
#define VFP_H

// #pragma once
// #include<iostream>
// #include<fstream>
// #include<math.h>
// #include<vector>
#include "../include/utility.h"

struct VfpResults {
    vector<double> y;//f(q,p) at the final time
    vector<double> t; // time vector where the solution is evaluated
    vector<double> density; // density wrt q over the time interval t
    vector<double> mass;
    vector<double> Pthz; // THz power over the time interval t
    //add Pthz
};

VfpResults integ_vfp(double *U0,double *H, double t1, double t2, double I,
                    double alpha_unfold,
                    int nt, int ntint, int nq,
                    int np, double Lq, double Lp, double grand_L);

#endif //VFP_H