#ifndef VFP_H
#define VFP_H


#pragma once
#include<iostream>
#include<fstream>
#include<math.h>
#include<vector>
#include "../include/utility.h"

struct VfpResults {
    std::vector<double> y;//f(q,p) at the final time
    std::vector<double> t; // time vector where the solution is evaluated
    std::vector<double> density; // density wrt q over the time interval t
    std::vector<double> mass; 
};

VfpResults integ_vfp(double t1, double t2, int nt, int ntint);

#endif //VFP_H