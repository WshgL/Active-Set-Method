#ifndef LEAST_SQUARES_ACTIVE_SET_HPP
#define LEAST_SQUARES_ACTIVE_SET_HPP

#include <Eigen/Dense>
#include <iostream>
#include <limits>
#include <cmath>

template <size_t N, size_t M>
class LeastSquaresActiveSet {
public:
    LeastSquaresActiveSet() = default;
    ~LeastSquaresActiveSet() = default;

    void init_solver(const size_t &num_step);

    void define_problem(const Eigen::Matrix<float, N, 1> &u_max,
                        const Eigen::Matrix<float, N, 1> &u_min,
                        const Eigen::Matrix<float, N, 1> &u_init,
                        const Eigen::Matrix<float, M, 1> &b,
                        const Eigen::Matrix<float, M, N> &A);

    void iterate_update();

    Eigen::Matrix<float, N, 1> get_result();

private:
    size_t _num_step{50};

    // 1 = active, upper bound; -1 = active, lower bound; 0 = free
    Eigen::Matrix<int, N, 1> _w_act;

    Eigen::Matrix<float, N, 1> _u_max;
    Eigen::Matrix<float, N, 1> _u_min;
    Eigen::Matrix<float, N, 1> _u;

    Eigen::Matrix<float, M, 1> _b;
    Eigen::Matrix<float, M, N> _A;
    Eigen::Matrix<float, M, 1> _d;
};

template <size_t N, size_t M>
void LeastSquaresActiveSet<N, M>::init_solver(const size_t &num_step) {
    _num_step = num_step;
    _w_act.setZero(); // Initialize all variables as free
}

template <size_t N, size_t M>
void LeastSquaresActiveSet<N, M>::define_problem(const Eigen::Matrix<float, N, 1> &u_max,
                                                 const Eigen::Matrix<float, N, 1> &u_min,
                                                 const Eigen::Matrix<float, N, 1> &u_init,
                                                 const Eigen::Matrix<float, M, 1> &b,
                                                 const Eigen::Matrix<float, M, N> &A) {
    _u_max = u_max;
    _u_min = u_min;

    _u = u_init.cwiseMax(u_min).cwiseMin(u_max);

    _b = b;
    _A = A;

    _d = _b - _A * _u;
}

template <size_t N, size_t M>
void LeastSquaresActiveSet<N, M>::iterate_update() {
    for (size_t iter = 0; iter < _num_step; iter++) {
        Eigen::Matrix<float, M, N> Aff = _A; // Copy of A
        Eigen::Matrix<double, M, N> Afd = _A.template cast<double>(); // Convert to double for better precision

        // Zero out columns corresponding to active variables
        for (size_t i = 0; i < N; i++) {
            if (_w_act(i) != 0) {
                Aff.col(i).setZero();
                Afd.col(i).setZero();
            }
        }

        // Compute pseudoinverse of Afd using SVD with tolerance
        Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> svd(Afd, Eigen::ComputeThinU | Eigen::ComputeThinV);
        double tolerance = std::numeric_limits<double>::epsilon() * std::max(Afd.cols(), Afd.rows()) * svd.singularValues().array().abs()(0);
        Eigen::VectorXd singularValues_inv = svd.singularValues();
        for (int i = 0; i < singularValues_inv.size(); ++i) {
            if (singularValues_inv(i) > tolerance) {
                singularValues_inv(i) = 1.0 / singularValues_inv(i);
            } else {
                singularValues_inv(i) = 0.0;
            }
        }
        Eigen::Matrix<double, N, M> Afd_inv = svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().transpose();
        Eigen::Matrix<float, N, M> Aff_inv = Afd_inv.template cast<float>();

        Eigen::Matrix<float, N, 1> p = Aff_inv * _d;

        // Zero out direction for active variables
        for (size_t i = 0; i < N; i++) {
            if (_w_act(i) != 0) {
                p(i) = 0.0f;
            }
        }

        Eigen::Matrix<float, N, 1> u_temp = _u + p;

        // Check feasibility and compute step size
        bool feasible = true;
        float ap_min = 1.0f;
        Eigen::Matrix<float, N, 1> ap = Eigen::Matrix<float, N, 1>::Ones();

        for (size_t i = 0; i < N; i++) {
            if (std::fabs(p(i)) > std::numeric_limits<float>::epsilon()) {
                if (u_temp(i) > _u_max(i)) {
                    feasible = false;
                    ap(i) = (_u_max(i) - _u(i)) / p(i);
                } else if (u_temp(i) < _u_min(i)) {
                    feasible = false;
                    ap(i) = (_u_min(i) - _u(i)) / p(i);
                }
                if (ap(i) < ap_min) {
                    ap_min = ap(i);
                }
            }
        }

        // Update u and d
        _u = _u + ap_min * p;
        _d = _d - Aff * (ap_min * p);

        if (!feasible) {
            for (size_t i = 0; i < N; i++) {
                if (std::fabs((_u(i) - _u_min(i))) < std::numeric_limits<float>::epsilon()) {
                    _w_act(i) = -1;
                    _u(i) = _u_min(i);
                } else if (std::fabs((_u(i) - _u_max(i))) < std::numeric_limits<float>::epsilon()) {
                    _w_act(i) = 1;
                    _u(i) = _u_max(i);
                }
            }
        } else {
            Eigen::Matrix<float, N, 1> la = _A.transpose() * _d;

            // Check optimality
            bool optimal = true;
            float la_min = 0.0f;

            for (size_t i = 0; i < N; i++) {
                if (_w_act(i) != 0) {
                    float lambda = la(i) * _w_act(i);
                    if (lambda < la_min) {
                        la_min = lambda;
                        optimal = false;
                    }
                }
            }

            if (optimal) {
                return; // Optimal solution found
            } else {
                for (size_t i = 0; i < N; i++) {
                    if (_w_act(i) != 0) {
                        float lambda = la(i) * _w_act(i);
                        if (std::fabs(lambda - la_min) < std::numeric_limits<float>::epsilon()) {
                            _w_act(i) = 0; // Release variable
                        }
                    }
                }
            }
        }
    }
}

template <size_t N, size_t M>
Eigen::Matrix<float, N, 1> LeastSquaresActiveSet<N, M>::get_result() {
    return _u;
}

#endif // LEAST_SQUARES_ACTIVE_SET_HPP
