#ifndef LEAST_SQUARES_ACTIVE_SET_HPP
#define LEAST_SQUARES_ACTIVE_SET_HPP

#include <iostream>
#include <Eigen/Dense>
#include <limits>
#include <cmath>
#include <cfloat>

template<typename MatrixType>
MatrixType computePseudoInverse(const MatrixType &A, double epsilon = std::numeric_limits<double>::epsilon())
{
    // Compute the singular value decomposition (SVD) of A
    Eigen::JacobiSVD<MatrixType> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = epsilon * std::max(A.cols(), A.rows()) * svd.singularValues().array().abs()(0);

    // Compute the pseudoinverse
    return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}

template <int N, int M>
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

    // 1 = active, upper bound; -1 = active, lower bound; 0 = free, unsaturated
    Eigen::Matrix<int, N, 1> _w_act;

    Eigen::Matrix<float, N, 1> _u_max;
    Eigen::Matrix<float, N, 1> _u_min;
    Eigen::Matrix<float, N, 1> _u;

    Eigen::Matrix<float, M, 1> _b;
    Eigen::Matrix<float, M, N> _A;
    Eigen::Matrix<float, M, 1> _d;
};

template <int N, int M>
void LeastSquaresActiveSet<N, M>::init_solver(const size_t &num_step) {
    _num_step = num_step;
    _w_act.setZero(); // free
}

template <int N, int M>
void LeastSquaresActiveSet<N, M>::define_problem(const Eigen::Matrix<float, N, 1> &u_max,
                                                 const Eigen::Matrix<float, N, 1> &u_min,
                                                 const Eigen::Matrix<float, N, 1> &u_init,
                                                 const Eigen::Matrix<float, M, 1> &b,
                                                 const Eigen::Matrix<float, M, N> &A) {
    _u_max = u_max;
    _u_min = u_min;

    // Constrain u_init between u_min and u_max
    _u = u_init.cwiseMax(u_min).cwiseMin(u_max);

    _b = b;
    _A = A;

    _d = b - A * _u;
}

template <int N, int M>
void LeastSquaresActiveSet<N, M>::iterate_update() {
    for (size_t iter = 0; iter < _num_step; iter++) {
        // Create copies of _A
        Eigen::Matrix<float, M, N> Aff = _A; // copy, M*N
        Eigen::Matrix<double, M, N> Afd = _A.cast<double>();

        // Zero out columns corresponding to active variables
        for (size_t i = 0; i < N; i++) { // column
            if (_w_act(i) == 1 || _w_act(i) == -1) { // active, saturated
                Aff.col(i).setZero();
                Afd.col(i).setZero();
            }
        }

        // Compute the pseudoinverse of Afd
        Eigen::Matrix<double, N, M> Afd_inv = computePseudoInverse(Afd);

        // Convert back to float
        Eigen::Matrix<float, N, M> Aff_inv = Afd_inv.cast<float>();

        Eigen::Matrix<float, N, 1> p = Aff_inv * _d;

        // Set zeros for the elements that are in active set
        for (size_t i = 0; i < N; i++) {
            if (_w_act(i) == 1 || _w_act(i) == -1) { // active, saturated
                p(i) = 0;
            }
        }

        Eigen::Matrix<float, N, 1> u_temp = _u + p;

        // Check feasibility
        bool feasible = true;
        float ap_min = 1; // alpha: 0 ~ 1
        Eigen::Matrix<float, N, 1> ap = Eigen::Matrix<float, N, 1>::Ones();

        for (size_t i = 0; i < N; i++) {
            if (fabsf(p(i)) > FLT_EPSILON) {
                if (u_temp(i) > _u_max(i)) { // infeasible, upper bound
                    feasible = false;
                    ap(i) = (_u_max(i) - _u(i)) / p(i);

                    if (ap(i) < ap_min) {
                        ap_min = ap(i);
                    }

                } else if (u_temp(i) < _u_min(i)) { // infeasible, lower bound
                    feasible = false;
                    ap(i) = (_u_min(i) - _u(i)) / p(i);

                    if (ap(i) < ap_min) {
                        ap_min = ap(i);
                    }

                } else {
                    ap(i) = 1;
                }

            } else {
                ap(i) = 1;
            }
        }

        // Update before checking optimality
        _d = _d - Aff * (ap_min * p);
        _u = _u + ap_min * p;

        if (!feasible) {
            for (size_t i = 0; i < N; i++) {
                if (fabsf(ap(i) - ap_min) < FLT_EPSILON) {
                    if (p(i) > 0) {
                        _w_act(i) = 1;
                    } else if (p(i) < 0) {
                        _w_act(i) = -1;
                    }
                }
            }

        } else {
            Eigen::Matrix<float, N, 1> la = _A.transpose() * _d; // ~dir, delta

            // Check optimality
            bool optimal = true;
            float la_min = 0;

            for (size_t i = 0; i < N; i++) {
                if (_w_act(i) == 0) { // free
                    la(i) = 0;

                } else if (_w_act(i) == 1 || _w_act(i) == -1) { // active, saturated
                    la(i) = la(i) * static_cast<float>(_w_act(i)); // lambda

                    if (la(i) < la_min) {
                        optimal = false;
                        la_min = la(i);
                    }
                }
            }

            if (!optimal) {
                for (size_t i = 0; i < N; i++) {
                    if (_w_act(i) != 0 && fabsf(la(i) - la_min) < FLT_EPSILON) {
                        _w_act(i) = 0; // free
                    }
                }

            } else {
                return; // Optimal solution found
            }
        }
    }
}

template <int N, int M>
Eigen::Matrix<float, N, 1> LeastSquaresActiveSet<N, M>::get_result() {
    return _u;
}

#endif // LEAST_SQUARES_ACTIVE_SET_HPP
