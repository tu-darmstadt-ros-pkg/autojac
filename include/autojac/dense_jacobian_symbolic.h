#pragma once

#include <cppad/example/cppad_eigen.hpp>

#include "autojac/types.h"

namespace autojac
{
/**
 * @brief The DenseJacobianSymbolic class handles the calculation of a dense jacobian of a function using symbolic evaluation.
 * @tparam Scalar scalar type used for representing the jacobian (i.e. double or float).
 */
template <typename Scalar>
class DenseJacobianSymbolic
{
public:
  /**
   * @brief Alias for std::shared_ptr
   */
  using Ptr = typename std::shared_ptr<DenseJacobianSymbolic<Scalar>>;

  /**
   * @brief Alias for std::shared_ptr using a const reference to the object
   */
  using ConstPtr = typename std::shared_ptr<const DenseJacobianSymbolic<Scalar>>;

  /**
   * @brief Constructor
   * @param function_input_dim the size of the input vector of the function which is equal to the number of columns of the jacobian matrix
   * @param function_output_dim the size of the output vector of the function which is equal to the number of rows of the jacobian matrix
   * @param function the function for which the jacobian should be calculated
   */
  DenseJacobianSymbolic(size_t function_input_dim, size_t function_output_dim, const CppAD::ADFun<Scalar>& function)
    : row_size_(function_output_dim)
    , col_size_(function_input_dim)
    , eigen_jac_vector_(function_input_dim * function_output_dim)

  {
    function_ = function;
  }

  /**
   * @brief Returns a dense Eigen map to access the jacobian.
   * @details The map needs just to be retrieved once.
   * @return dense Eigen map to access the jacobian
   */
  ConstDenseJacobianMap<Scalar> getMap() { return ConstDenseJacobianMap<Scalar>(eigen_jac_vector_.data(), row_size_, col_size_); }

  /**
   * @brief Calculate the jacobian for the given input vector
   * @param input_vector input vector for which the jacobian should be calculated
   */
  void updateJacobian(const VectorXS<Scalar>& input_vector)
  {
    VectorXS<Scalar> tmp;
    // as the funciton asignment uses move semantics the data pointer of maps generated before the last call of updateJacobian() would point to invalid data.
    tmp = function_.Jacobian(input_vector);
    memcpy(eigen_jac_vector_.data(), tmp.data(), row_size_ * col_size_ * sizeof(Scalar));
  }

private:
  const int row_size_;
  const int col_size_;
  VectorXS<Scalar> eigen_jac_vector_;
  CppAD::ADFun<Scalar> function_;
};

// precompile for most common use case
extern template class DenseJacobianSymbolic<double>;
}  // namespace autojac
