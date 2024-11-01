#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <cmath>
#include "least_squares_active_set.hpp"

int main() {
    const int N = 2;
    const int M = 3;

    // Define the matrices and vectors
    Eigen::Matrix<float, M, N> A;
    Eigen::Matrix<float, M, 1> b;
    Eigen::Matrix<float, N, 1> u_init;
    Eigen::Matrix<float, N, 1> u_min;
    Eigen::Matrix<float, N, 1> u_max;

    // Example values
    A << 1, 2,
         3, 4,
         5, 6;
    b << 7, 8, 9;
    u_init << 0, 0;
    u_min << -10, -10;
    u_max << 10, 10;

    // Create the solver
    LeastSquaresActiveSet<N, M> solver;
    solver.init_solver(100);
    solver.define_problem(u_max, u_min, u_init, b, A);
    solver.iterate_update();

    Eigen::Matrix<float, N, 1> u_result = solver.get_result();

    std::cout << "Result u:\n" << u_result << std::endl;

    // Verify the result
    Eigen::Matrix<float, M, 1> residual = b - A * u_result;
    float residual_norm = residual.norm();

    std::cout << "Residual norm: " << residual_norm << std::endl;

    return 0;
}
