#ifndef LINEAR_SOLVERS_HPP
#define LINEAR_SOLVERS_HPP

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// ============================================================================
//  linear_solvers.hpp
// ----------------------------------------------------------------------------
//  Declarations of every linear-system solver implemented in this folder.
//
//  Origin file                        Function (here)        (original name)
//  ---------------------------------------------------------------------------
//  gradient_method.cpp                gradientMethod          steepestDescent
//  conjugate_gradient.cpp             conjugateGradient       steepestDescent *
//  precon_conjugate_gradient.cpp      preconditionedCG        preconditionedCG
//  preconditioned_richardson.cpp      preconditionedRichardson (same name)
//  stat_richardson.cpp                stationaryRichardson    inline in main()
//  jacobi.cpp                         jacobi                  inline in main()
//  gs.cpp                             gaussSeidel             inline in main()
//
//  * NOTE: the original conjugate_gradient.cpp also names its function
//    "steepestDescent", which collides with the one in gradient_method.cpp.
//    Since a header can only declare one function per signature, the two are
//    renamed here according to what they actually compute.
//
//  All functions take the system matrix A and right-hand side b by const
//  reference (no copies) and return the computed solution as an Eigen::VectorXd.
// ============================================================================

// Gaussian/Steepest-Descent method (from gradient_method.cpp).
// Assumes A is symmetric positive definite.
Eigen::VectorXd gradientMethod(const Eigen::MatrixXd& A,
                               const Eigen::VectorXd& b,
                               double tol,
                               int n);

// Conjugate-Gradient method (from conjugate_gradient.cpp).
// Assumes A is symmetric positive definite.
Eigen::VectorXd conjugateGradient(const Eigen::MatrixXd& A,
                                  const Eigen::VectorXd& b,
                                  double tol,
                                  int n);

// Preconditioned Conjugate-Gradient method (from precon_conjugate_gradient.cpp).
// P is the preconditioner (e.g. Jacobi: diagonal of A).
Eigen::VectorXd preconditionedCG(const Eigen::MatrixXd& A,
                                 const Eigen::VectorXd& b,
                                 double tol,
                                 int n,
                                 const Eigen::MatrixXd& P);

// Stationary Richardson method (from stat_richardson.cpp).
// The optimal relaxation parameter alpha = 2/(lambda_max + lambda_min)
// is computed internally from the spectrum of A (SelfAdjointEigenSolver).
Eigen::VectorXd stationaryRichardson(const Eigen::MatrixXd& A,
                                     const Eigen::VectorXd& b,
                                     const Eigen::VectorXd& x0,
                                     int maxIterations);

// Preconditioned Richardson method (from preconditioned_richardson.cpp).
// alpha is derived from the eigenvalues of P^{-1}A.
Eigen::VectorXd preconditionedRichardson(const Eigen::MatrixXd& A,
                                         const Eigen::VectorXd& b,
                                         const Eigen::MatrixXd& P,
                                         double tol,
                                         int n);

// Jacobi iterative method (from jacobi.cpp).
Eigen::VectorXd jacobi(const Eigen::MatrixXd& A,
                       const Eigen::VectorXd& b,
                       const Eigen::VectorXd& x0,
                       int maxIterations);

// Gauss-Seidel iterative method (from gs.cpp).
Eigen::VectorXd gaussSeidel(const Eigen::MatrixXd& A,
                            const Eigen::VectorXd& b,
                            const Eigen::VectorXd& x0,
                            int maxIterations);

// Helper (not in the originals; extracted from the repeated test-main code):
// builds the 1D Laplacian tridiagonal matrix used by most solvers.
Eigen::MatrixXd build1DLaplacian(int n);

#endif // LINEAR_SOLVERS_HPP