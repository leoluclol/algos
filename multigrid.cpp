#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <cmath>
#include "linear_solvers.h"

// da fare:
// filo di lunghezza L
// divisione in h segmenti

using namespace Eigen;

int main(){
	int h = 512;
	int L = 10; // 10m di filo
	VectorXd force = VectorXd::Ones(h);
	VectorXd displacements = VectorXd::Zeros(h);
	
	
	
	// 1. Programmatically build a 1D Laplacian matrix
	MatrixXd A = MatrixXd::Zero(h, h);
    	for (int i = 0; i < h; i++) {
    		A(i, i) = 2.0;
        	if (i > 0) A(i, i - 1) = -1.0;
       		if (i < n - 1) A(i, i + 1) = -1.0;
    	}

	// A * displacements = force
	
	displacements = conjugateGradient(A, force, h, 1e-10);
		
}	

