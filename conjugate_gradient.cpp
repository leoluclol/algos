#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> // Required for eigenvalue computations

using namespace Eigen; // Correct namespace syntax

// Pass matrices and vectors by const reference to avoid expensive copying
VectorXd steepestDescent(const MatrixXd& A, const VectorXd& b, double tol, int n) {
    VectorXd x = VectorXd::Zero(n);
    VectorXd r = b - A * x; // Initialize residual correctly
    VectorXd d = r;

    VectorXd Ad;
    double alpha;
    double beta;
    int k = 0;

    // Standard stopping criterion for iterative solvers: residual norm
    while (r.norm() > tol) {
        Ad = A * d; //precalcolo Ad per non sprecare compute
        alpha = d.dot(r)/d.dot(Ad); //calcola alfa ottimale
        x = x + alpha * d; //fai un passo in direzione d
        r = r - alpha * Ad; //aggiorna residuo
	beta = (Ad.dot(r))/(Ad.dot(d)); //calcola beta ottimale per rendere la nuova d perpendicolare alla d di prima nello spazio generato da A
					//d_(k+1) si calcola correggendo r_(k+1) con un fattore r_k*beta
	d = r - beta * d; //calcola d_(k+1) come sopra
        k = k + 1;
    }

    std::cout << k << "\n";
    return x;
}

int main() {
    int n = 50; // Increasing N makes the problem drastically harder
    
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

    // 4. Call the solver
    double tol = 1e-10;
    std::cout << "Solving system of size " << n << "x" << n << "...\n";
    VectorXd result = steepestDescent(A, b, tol, n);

    // Calculate the error to prove it found the right answer
    double error = (result - sol).norm();
    std::cout << "Final Error (L2 Norm): " << error << std::endl;

    return 0;
}
