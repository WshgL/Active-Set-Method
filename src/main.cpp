#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <cmath>
#include "least_squares_active_set.hpp"

int main() {
    // Test Case 1: Basic Test
    {
        const size_t N = 3;
        const size_t M = 2;
        std::cout << "Test Case 1: Basic Test\n";

        LeastSquaresActiveSet<N, M> solver;
        solver.init_solver(200);

        Eigen::Matrix<float, N, 1> u_max;
        u_max << 5, 5, 5;
        Eigen::Matrix<float, N, 1> u_min;
        u_min << 0, 0, 0;
        Eigen::Matrix<float, N, 1> u_init;
        u_init << 1, 1, 1;
        Eigen::Matrix<float, M, 1> b;
        b << 7, 8;
        Eigen::Matrix<float, M, N> A;
        A << 1, 2, 3,
             4, 5, 6;

        solver.define_problem(u_max, u_min, u_init, b, A);
        solver.iterate_update();

        Eigen::Matrix<float, N, 1> result = solver.get_result();

        // Verify and output results
        std::cout << "Optimal solution u:\n" << result.transpose() << std::endl;
        bool constraints_satisfied = ((result.array() >= u_min.array()).all() && (result.array() <= u_max.array()).all());
        std::cout << "u_min <= u <= u_max: " << std::boolalpha << constraints_satisfied << std::endl;
        Eigen::Matrix<float, M, 1> residual = A * result - b;
        std::cout << "Residual norm ||A*u - b||: " << residual.norm() << "\n\n";
    }

    // Test Case 2: Tightened Constraints
    {
        const size_t N = 3;
        const size_t M = 2;
        std::cout << "Test Case 2: Tightened Constraints\n";

        LeastSquaresActiveSet<N, M> solver;
        solver.init_solver(200);

        Eigen::Matrix<float, N, 1> u_max;
        u_max << 2, 2, 2;
        Eigen::Matrix<float, N, 1> u_min;
        u_min << 0.5, 0.5, 0.5;
        Eigen::Matrix<float, N, 1> u_init;
        u_init << 1, 1, 1;
        Eigen::Matrix<float, M, 1> b;
        b << 7, 8;
        Eigen::Matrix<float, M, N> A;
        A << 1, 2, 3,
             4, 5, 6;

        solver.define_problem(u_max, u_min, u_init, b, A);
        solver.iterate_update();

        Eigen::Matrix<float, N, 1> result = solver.get_result();

        // Verify and output results
        std::cout << "Optimal solution u:\n" << result.transpose() << std::endl;
        bool constraints_satisfied = ((result.array() >= u_min.array()).all() && (result.array() <= u_max.array()).all());
        std::cout << "u_min <= u <= u_max: " << std::boolalpha << constraints_satisfied << std::endl;
        Eigen::Matrix<float, M, 1> residual = A * result - b;
        std::cout << "Residual norm ||A*u - b||: " << residual.norm() << "\n\n";
    }

    // Test Case 3: Overdetermined System
    {
        const size_t N = 2;
        const size_t M = 4;
        std::cout << "Test Case 3: Overdetermined System\n";

        LeastSquaresActiveSet<N, M> solver;
        solver.init_solver(200);

        Eigen::Matrix<float, N, 1> u_max;
        u_max << 10, 10;
        Eigen::Matrix<float, N, 1> u_min;
        u_min << -10, -10;
        Eigen::Matrix<float, N, 1> u_init;
        u_init << 0, 0;
        Eigen::Matrix<float, M, 1> b;
        b << 2, 3, 5, 7;
        Eigen::Matrix<float, M, N> A;
        A << 1, 2,
             2, 1,
             3, 4,
             4, 3;

        solver.define_problem(u_max, u_min, u_init, b, A);
        solver.iterate_update();

        Eigen::Matrix<float, N, 1> result = solver.get_result();

        // Verify and output results
        std::cout << "Optimal solution u:\n" << result.transpose() << std::endl;
        bool constraints_satisfied = ((result.array() >= u_min.array()).all() && (result.array() <= u_max.array()).all());
        std::cout << "u_min <= u <= u_max: " << std::boolalpha << constraints_satisfied << std::endl;
        Eigen::Matrix<float, M, 1> residual = A * result - b;
        std::cout << "Residual norm ||A*u - b||: " << residual.norm() << "\n\n";
    }

    // Test Case 4: Underdetermined System
    {
        const size_t N = 4;
        const size_t M = 2;
        std::cout << "Test Case 4: Underdetermined System\n";

        LeastSquaresActiveSet<N, M> solver;
        solver.init_solver(200);

        Eigen::Matrix<float, N, 1> u_max;
        u_max << 5, 5, 5, 5;
        Eigen::Matrix<float, N, 1> u_min;
        u_min << 1, 1, 1, 1;
        Eigen::Matrix<float, N, 1> u_init;
        u_init << 2, 2, 2, 2;
        Eigen::Matrix<float, M, 1> b;
        b << 10, 14;
        Eigen::Matrix<float, M, N> A;
        A << 1, 2, 1, 2,
             2, 1, 2, 1;

        solver.define_problem(u_max, u_min, u_init, b, A);
        solver.iterate_update();

        Eigen::Matrix<float, N, 1> result = solver.get_result();

        // Verify and output results
        std::cout << "Optimal solution u:\n" << result.transpose() << std::endl;
        bool constraints_satisfied = ((result.array() >= u_min.array()).all() && (result.array() <= u_max.array()).all());
        std::cout << "u_min <= u <= u_max: " << std::boolalpha << constraints_satisfied << std::endl;
        Eigen::Matrix<float, M, 1> residual = A * result - b;
        std::cout << "Residual norm ||A*u - b||: " << residual.norm() << "\n\n";
    }

    // Test Case 5: Upper Bound Only
    {
        const size_t N = 3;
        const size_t M = 3;
        std::cout << "Test Case 5: Upper Bound Only\n";

        LeastSquaresActiveSet<N, M> solver;
        solver.init_solver(200);

        Eigen::Matrix<float, N, 1> u_max;
        u_max << 2, 2, 2;
        Eigen::Matrix<float, N, 1> u_min;
        u_min.setConstant(-std::numeric_limits<float>::infinity());
        Eigen::Matrix<float, N, 1> u_init;
        u_init << 0, 0, 0;
        Eigen::Matrix<float, M, 1> b;
        b << 1, 2, 3;
        Eigen::Matrix<float, M, N> A = Eigen::Matrix<float, M, N>::Identity();

        solver.define_problem(u_max, u_min, u_init, b, A);
        solver.iterate_update();

        Eigen::Matrix<float, N, 1> result = solver.get_result();

        // Verify and output results
        std::cout << "Optimal solution u:\n" << result.transpose() << std::endl;
        bool constraints_satisfied = ((result.array() >= u_min.array()).all() && (result.array() <= u_max.array()).all());
        std::cout << "u_min <= u <= u_max: " << std::boolalpha << constraints_satisfied << std::endl;
        Eigen::Matrix<float, M, 1> residual = A * result - b;
        std::cout << "Residual norm ||A*u - b||: " << residual.norm() << "\n\n";
    }

    // Test Case 6: Lower Bound Only
    {
        const size_t N = 3;
        const size_t M = 3;
        std::cout << "Test Case 6: Lower Bound Only\n";

        LeastSquaresActiveSet<N, M> solver;
        solver.init_solver(200);

        Eigen::Matrix<float, N, 1> u_max;
        u_max.setConstant(std::numeric_limits<float>::infinity());
        Eigen::Matrix<float, N, 1> u_min;
        u_min << 1, 1, 1;
        Eigen::Matrix<float, N, 1> u_init;
        u_init << 0, 0, 0;
        Eigen::Matrix<float, M, 1> b;
        b << 1, 2, 3;
        Eigen::Matrix<float, M, N> A = Eigen::Matrix<float, M, N>::Identity();

        solver.define_problem(u_max, u_min, u_init, b, A);
        solver.iterate_update();

        Eigen::Matrix<float, N, 1> result = solver.get_result();

        // Verify and output results
        std::cout << "Optimal solution u:\n" << result.transpose() << std::endl;
        bool constraints_satisfied = ((result.array() >= u_min.array()).all() && (result.array() <= u_max.array()).all());
        std::cout << "u_min <= u <= u_max: " << std::boolalpha << constraints_satisfied << std::endl;
        Eigen::Matrix<float, M, 1> residual = A * result - b;
        std::cout << "Residual norm ||A*u - b||: " << residual.norm() << "\n\n";
    }

    // Test Case 7: No Constraints
    {
        const size_t N = 3;
        const size_t M = 2;
        std::cout << "Test Case 7: No Constraints\n";

        LeastSquaresActiveSet<N, M> solver;
        solver.init_solver(200);

        Eigen::Matrix<float, N, 1> u_max;
        u_max.setConstant(std::numeric_limits<float>::infinity());
        Eigen::Matrix<float, N, 1> u_min;
        u_min.setConstant(-std::numeric_limits<float>::infinity());
        Eigen::Matrix<float, N, 1> u_init;
        u_init << 0, 0, 0;
        Eigen::Matrix<float, M, 1> b;
        b << 7, 8;
        Eigen::Matrix<float, M, N> A;
        A << 1, 2, 3,
             4, 5, 6;

        solver.define_problem(u_max, u_min, u_init, b, A);
        solver.iterate_update();

        Eigen::Matrix<float, N, 1> result = solver.get_result();

        // Verify and output results
        std::cout << "Optimal solution u:\n" << result.transpose() << std::endl;
        bool constraints_satisfied = ((result.array() >= u_min.array()).all() && (result.array() <= u_max.array()).all());
        std::cout << "u_min <= u <= u_max: " << std::boolalpha << constraints_satisfied << std::endl;
        Eigen::Matrix<float, M, 1> residual = A * result - b;
        std::cout << "Residual norm ||A*u - b||: " << residual.norm() << "\n\n";
    }

    // Test Case 8: Negative Variables
    {
        const size_t N = 2;
        const size_t M = 2;
        std::cout << "Test Case 8: Negative Variables\n";

        LeastSquaresActiveSet<N, M> solver;
        solver.init_solver(200);

        Eigen::Matrix<float, N, 1> u_max;
        u_max << 0, 0;
        Eigen::Matrix<float, N, 1> u_min;
        u_min << -5, -5;
        Eigen::Matrix<float, N, 1> u_init;
        u_init << -1, -1;
        Eigen::Matrix<float, M, 1> b;
        b << -3, -4;
        Eigen::Matrix<float, M, N> A;
        A << 1, 2,
             3, 4;

        solver.define_problem(u_max, u_min, u_init, b, A);
        solver.iterate_update();

        Eigen::Matrix<float, N, 1> result = solver.get_result();

        // Verify and output results
        std::cout << "Optimal solution u:\n" << result.transpose() << std::endl;
        bool constraints_satisfied = ((result.array() >= u_min.array()).all() && (result.array() <= u_max.array()).all());
        std::cout << "u_min <= u <= u_max: " << std::boolalpha << constraints_satisfied << std::endl;
        Eigen::Matrix<float, M, 1> residual = A * result - b;
        std::cout << "Residual norm ||A*u - b||: " << residual.norm() << "\n\n";
    }

    return 0;
}
