#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> // Required for eigenvalue computations

using namespace Eigen; // Correct namespace syntax

// Pass matrices and vectors by const reference to avoid expensive copying
VectorXd steepestDescent(const MatrixXd& A, const VectorXd& b, double tol, int n) {
    VectorXd x = VectorXd::Zero(n);
    VectorXd r = b - A * x; // Initialize residual correctly

    VectorXd Ar; 
    double alpha;
    int k = 0;

    // Standard stopping criterion for iterative solvers: residual norm
    while (r.norm() > tol) {
        Ar = A * r;
        alpha = r.dot(r)/r.dot(Ar);
        x = x + alpha * r;
        r = r - alpha * Ar;
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
