// ============================================================================
//  linear_solvers.cpp
// ----------------------------------------------------------------------------
//  Implementations of every linear-system solver declared in linear_solvers.hpp.
//
//  Each function mirrors (line by line) the algorithm in the matching source
//  file in this folder. The only deviations from the originals:
//    * the two functions originally both named "steepestDescent" got distinct
//      names (gradientMethod / conjugateGradient);
//    * the algorithms that lived inline inside main() (stationary Richardson,
//      Jacobi, Gauss-Seidel) were extracted into standalone functions;
//    * debug prints of the iteration count were removed so the functions are
//      side-effect free and reusable.
// ============================================================================

#include "linear_solvers.hpp"

// ----------------------------------------------------------------------------
// Gradient (Steepest Descent) method  —  original: gradient_method.cpp
// ----------------------------------------------------------------------------
Eigen::VectorXd gradientMethod(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                               double tol, int n) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd r = b - A * x;   // initial residual

    Eigen::VectorXd Ar;
    double alpha;
    int k = 0;

    // Standard stopping criterion for iterative solvers: residual norm
    while (r.norm() > tol) {
        Ar = A * r;
        alpha = r.dot(r) / r.dot(Ar);
        x = x + alpha * r;
        r = r - alpha * Ar;
        k = k + 1;
    }
    return x;
}

// ----------------------------------------------------------------------------
// Conjugate Gradient method  —  original: conjugate_gradient.cpp
// ----------------------------------------------------------------------------
Eigen::VectorXd conjugateGradient(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                  double tol, int n) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd r = b - A * x;   // initial residual
    Eigen::VectorXd d = r;

    Eigen::VectorXd Ad;
    double alpha;
    double beta;
    int k = 0;

    // Standard stopping criterion for iterative solvers: residual norm
    while (r.norm() > tol) {
        Ad = A * d;                          // precompute Ad to avoid wasted work
        alpha = d.dot(r) / d.dot(Ad);        // optimal alpha
        x = x + alpha * d;                   // step in direction d
        r = r - alpha * Ad;                  // update residual
        beta = (Ad.dot(r)) / (Ad.dot(d));    // optimal beta so the new d is
                                             // A-orthogonal to the previous d
        d = r - beta * d;                    // compute d_{k+1}
        k = k + 1;
    }
    return x;
}

// ----------------------------------------------------------------------------
// Preconditioned Conjugate Gradient  —  original: precon_conjugate_gradient.cpp
// ----------------------------------------------------------------------------
Eigen::VectorXd preconditionedCG(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                 double tol, int n, const Eigen::MatrixXd& P) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd r = b - A * x;        // initial residual
    Eigen::VectorXd z = P.inverse() * r;  // apply preconditioner
    Eigen::VectorXd d = z;

    Eigen::VectorXd Ad;
    double alpha;
    double beta;
    int k = 0;

    // Standard stopping criterion for iterative solvers: residual norm
    while (r.norm() > tol) {
        Ad = A * d;
        alpha = z.dot(r) / d.dot(Ad);
        x = x + alpha * d;
        r = r - alpha * Ad;
        z = P.inverse() * r;
        beta = (Ad.dot(z)) / (Ad.dot(d));
        d = z - beta * d;
        k = k + 1;
    }
    return x;
}

// ----------------------------------------------------------------------------
// Stationary Richardson method  —  original: stat_richardson.cpp
// ----------------------------------------------------------------------------
Eigen::VectorXd stationaryRichardson(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                     const Eigen::VectorXd& x0, int maxIterations) {
    Eigen::VectorXd x = x0;
    int k = 0;

    // Compute alpha_opt ONCE, outside the loop, from the spectrum of A.
    // SelfAdjointEigenSolver is optimized for symmetric matrices.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(A);
    double lambda_min = eigensolver.eigenvalues().minCoeff();
    double lambda_max = eigensolver.eigenvalues().maxCoeff();
    double alpha_opt = 2.0 / (lambda_max + lambda_min);

    Eigen::VectorXd r;
    while (k < maxIterations) {
        r = b - A * x;
        x = x + alpha_opt * r;
        k = k + 1;
    }
    return x;
}

// ----------------------------------------------------------------------------
// Preconditioned Richardson method  —  original: preconditioned_richardson.cpp
// ----------------------------------------------------------------------------
Eigen::VectorXd preconditionedRichardson(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                         const Eigen::MatrixXd& P, double tol, int n) {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd r = b - A * x;   // initial residual

    // Compute alpha from the eigenvalues of P^{-1}A (EigenSolver)
    Eigen::EigenSolver<Eigen::MatrixXd> es(P.inverse() * A);
    double lambda_min = es.eigenvalues().real().minCoeff();
    double lambda_max = es.eigenvalues().real().maxCoeff();
    double alpha = 2.0 / (lambda_min + lambda_max);
    int k = 0;

    // Standard stopping criterion for iterative solvers: residual norm
    while (r.norm() > tol) {
        // P.inverse() * r is mathematically fine; for larger systems
        // P.colPivHouseholderQr().solve(r) is safer/faster.
        Eigen::VectorXd z = P.inverse() * r;
        x = x + alpha * z;
        r = b - A * x;   // update residual
        k = k + 1;
    }
    return x;
}

// ----------------------------------------------------------------------------
// Jacobi method  —  original: jacobi.cpp (algorithm inside main())
// ----------------------------------------------------------------------------
Eigen::VectorXd jacobi(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                       const Eigen::VectorXd& x0, int maxIterations) {
    int n = static_cast<int>(A.rows());
    Eigen::VectorXd x_old = x0;
    Eigen::VectorXd x_new = x0;

    int k = 0;
    while (k < maxIterations) {
        x_old = x_new;

        for (int i = 0; i < n; ++i) {   // iterate on rows
            double total = 0.0;

            for (int j = 0; j < n; ++j) {   // iterate on the elements
                if (i != j) {
                    total += x_old[j] * A(i, j);
                }
            }
            x_new[i] = (b[i] - total) / A(i, i);
        }
        ++k;
    }
    return x_new;
}

// ----------------------------------------------------------------------------
// Gauss-Seidel method  —  original: gs.cpp (algorithm inside main())
// ----------------------------------------------------------------------------
Eigen::VectorXd gaussSeidel(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& x0, int maxIterations) {
    int n = static_cast<int>(A.rows());
    Eigen::VectorXd x_old = x0;
    Eigen::VectorXd x_new = x0;

    int k = 0;
    while (k < maxIterations) {
        x_old = x_new;

        for (int i = 0; i < n; ++i) {   // iterate on rows
            double total = 0.0;

            for (int j = 0; j < n; ++j) {   // iterate on the elements
                if (j > i) {
                    total += x_old[j] * A(i, j);
                } else if (j < i) {
                    total += x_new[j] * A(i, j);
                }
            }
            x_new[i] = (b[i] - total) / A(i, i);
        }
        ++k;
    }
    return x_new;
}

// ----------------------------------------------------------------------------
// Helper: 1D Laplacian tridiagonal matrix (dup. test-main code in 4 files)
// ----------------------------------------------------------------------------
Eigen::MatrixXd build1DLaplacian(int n) {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
    for (int i = 0; i < n; i++) {
        A(i, i) = 2.0;
        if (i > 0) A(i, i - 1) = -1.0;
        if (i < n - 1) A(i, i + 1) = -1.0;
    }
    return A;
}