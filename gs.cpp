#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>

int main() {
    srand(42);

    int n = 5;

    Eigen::VectorXd x_original = Eigen::VectorXd::Random(n);

    Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
    A += Eigen::MatrixXd::Identity(n, n) * 5.0; // make A diagonally dominant
    Eigen::VectorXd b = A * x_original;
    Eigen::VectorXd x_0 = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd x_1 = x_0;

    std::cout << "\n" << x_0 << "\n\n" << x_1 << "\n";

    std::cout << "Starting value of array:\n" << x_0 << "\n";

    // Changed to double to hold the floating point values from VectorXd 
    // and avoid narrowing compilation errors
    double total;
    int k = 0;

    std::cout << "Value of A \n" << A << "\n";
    std::cout << "Value of actual solution\n" << x_original << "\n";

    double error;
    while (k < 20) {
        x_0 = x_1;

        for (int i = 0; i < n; ++i){ // Iterate on rows
            total = 0;

            for (int j = 0; j < n; ++j) { // Iterate on the elements
                if (j>i) {
                    total += x_0[j] * A(i, j);
                }else if (j<i){
                    total += x_1[j] * A(i, j);
                }   
            }
            x_1[i] = (b[i] - total) / A(i, i);
        }   
        ++k;

        //if(k%10 == 0){
            error = 0;
            for (int x = 0; x < n; ++x){
                error = error + x_1[x] * x_1[x];
            }
            std::cout << "Error after " << k << " iterations: " << error << "\n";
        //}
        
    }

    return 0;
}