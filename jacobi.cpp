#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>

int main() {
    srand((unsigned int) time(0));

    int n = 5;

    Eigen::VectorXd x_original = Eigen::VectorXd::Random(n);

    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    A += Eigen::MatrixXd::Identity(n, n) * 5.0; // make A diagonally dominant
    Eigen::VectorXd b = A * x_original;
    Eigen::VectorXd x_0 = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd x_1 = x_0;

    std::cout << "Starting value of array:\n" << x_0 << "\n";

    // Changed to double to hold the floating point values from VectorXd 
    // and avoid narrowing compilation errors
    double total;
    int k = 0;

    std::cout << "Value of A \n" << A << "\n";
    std::cout << "Value of actual solution\n" << x_original << "\n";

    while (k < 100) {
        x_0 = x_1;

        for (int i = 0; i < n; ++i){ // Iterate on rows
            total = 0;

            for (int j = 0; j < n; ++j) { // Iterate on the elements
                if (i != j) {
                    total += x_0[j] * A(i, j);
                }
            }
            x_1[i] = (b[i] - total) / A(i, i);
        }   
        ++k;
    }
    std::cout << "Value of proposed solution after " << k << " iterations: \n" << x_1 << "\n";

    return 0;
}