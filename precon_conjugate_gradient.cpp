#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> // Required for eigenvalue computations

using namespace Eigen; // Correct namespace syntax

// Pass matrices and vectors by const reference to avoid expensive copying
VectorXd preconditionedCG(const MatrixXd& A, const VectorXd& b, double tol, int n, const MatrixXd & P) {
    VectorXd x = VectorXd::Zero(n);
    VectorXd r = b - A * x; // Initialize residual correctly
    VectorXd z = P.inverse()*r;
    VectorXd d = z;
    VectorXd Ad;
    double alpha;
    double beta;
    int k = 0;

    // Standard stopping criterion for iterative solvers: residual norm
    while (r.norm() > tol) {
        Ad = A * d; //precalcolo Ad per non sprecare compute
        alpha = z.dot(r)/d.dot(Ad); //calcola alfa ottimale
        x = x + alpha * d; //fai un passo in direzione d
        r = r - alpha * Ad; //aggiorna residuo
	z = P.inverse() * r;
	beta = (Ad.dot(z))/(Ad.dot(d)); //calcola beta ottimale per rendere la nuova d perpendicolare alla d di prima nello spazio generato da A
					//d_(k+1) si calcola correggendo r_(k+1) con un fattore r_k*beta
	d = z - beta * d; //calcola d_(k+1) come sopra
        k = k + 1;
    }

    std::cout << k << "\n";
    return x;
}

int main() {
    int n;
    std::cout << "enter n\n";
    std::cin >>  n;
    
    // 1. Programmatically build a 1D Laplacian matrix
    MatrixXd A = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; i++) {
        A(i, i) = 2.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
    } 
   
    // 2. Define the exact solution (a vector of all 1s)
    VectorXd sol = VectorXd::Ones(n); 

    // 3. Calculate b
    VectorXd b = A * sol;

    MatrixXd P = MatrixXd::Zero(n, n);
	for(int i = 0; i < n; i++) {
    	P(i, i) = A(i, i);
	}

    // 4. Call the solver
    double tol = 1e-10;
    std::cout << "Solving system of size " << n << "x" << n << "...\n";
    VectorXd result = preconditionedCG(A, b, tol, n, P);

    // Calculate the error to prove it found the right answer
    double error = (result - sol).norm();
    std::cout << "Final Error (L2 Norm): " << error << std::endl;

    return 0;
}
