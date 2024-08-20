#pragma once

#include "Eigen/SparseCore"

namespace autojac
{
/**
 * @brief Read only map for a sparse row-major jacobian
 * @tparam Scalar scalar type used for the matrix entries
 */
template <typename Scalar>
using ConstSparseJacobianMap = Eigen::Map<const Eigen::SparseMatrix<Scalar, Eigen::RowMajor>>;

/**
 * @brief Read only map for a dense row-major jacobian
 * @tparam Scalar scalar type used for the matrix entries
 */
template <typename Scalar>
using ConstDenseJacobianMap = Eigen::Map<const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

/**
 * @brief Dynamic column vector for generic entry types
 * @tparam Scalar scalar type used for the vector entries
 */
template <typename Scalar>
using VectorXS = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

/**
 * @brief Dynamic dense matrix for generic entry types
 * @tparam Scalar scalar type used for the matrix entries
 */
template <typename Scalar>
using MatrixXS = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
}  // namespace autojac
