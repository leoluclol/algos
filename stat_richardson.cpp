#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues> // Necessario per il calcolo degli autovalori
#include <cstdlib>

int main() {
    srand(42);

    int n = 5;

    Eigen::VectorXd x_original = Eigen::VectorXd::Random(n);

    // 1. Generiamo una matrice Simmetrica Definita Positiva (SDP)
    // affinché gli autovalori siano strettamente reali e positivi.
    Eigen::MatrixXd A_rand = Eigen::MatrixXd::Random(n, n);
    Eigen::MatrixXd A = A_rand.transpose() * A_rand; 

    Eigen::VectorXd b = A * x_original;
    Eigen::VectorXd x = Eigen::VectorXd::Ones(n);
    
    int k = 0;

    std::cout << "Value of A \n" << A << "\n\n";
    std::cout << "Value of actual solution\n" << x_original << "\n\n";

    // 2. Calcoliamo gli autovalori usando SelfAdjointEigenSolver 
    // (ottimizzato per matrici simmetriche)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(A);
    if (eigensolver.info() != Eigen::Success) {
        std::cerr << "Calcolo degli autovalori fallito!" << std::endl;
        return 1;
    }

    // Estraiamo il massimo e il minimo
    double lambda_min = eigensolver.eigenvalues().minCoeff();
    double lambda_max = eigensolver.eigenvalues().maxCoeff();

    // 3. Calcoliamo alpha_opt UNA SOLA VOLTA fuori dal ciclo
    double alpha_opt = 2.0 / (lambda_max + lambda_min);

    Eigen::VectorXd r;

    while (k < 3000) {
        r = b - A * x;
        x = x + alpha_opt * r;
        
        k++; // 4. Incremento per evitare il loop infinito
    }
    
    std::cout << "Value of solution found (after 3000 iterations)\n" << x << "\n";    

    return 0;
}