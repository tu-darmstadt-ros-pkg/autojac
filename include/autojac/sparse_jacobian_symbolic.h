#pragma once

#include <cppad/example/cppad_eigen.hpp>

#include "autojac/types.h"
#include "autojac/sparsity_pattern_handler.h"

namespace autojac
{
/**
 * @brief The SparseJacobianSymbolic class handles the calculation of a sparse jacobian of a function using symbolic evaluation.
 * @tparam Scalar scalar type used for representing the jacobian (i.e. double or float).
 */
template <typename Scalar>
class SparseJacobianSymbolic
{
public:
  /**
   * @brief Alias for std::shared_ptr
   */
  using Ptr = typename std::shared_ptr<SparseJacobianSymbolic<Scalar>>;

  /**
   * @brief Alias for std::shared_ptr using a const reference to the object
   */
  using ConstPtr = typename std::shared_ptr<const SparseJacobianSymbolic<Scalar>>;

  /**
   * @brief Construct an object handling the automatic differentiation of a CppAD function that also offers an Eigen compatible representation.
   * @param function_input_dim the size of the input vector of the function which is equal to the number of columns of the jacobian matrix
   * @param function_output_dim the size of the output vector of the function which is equal to the number of rows of the jacobian matrix
   * @param function the function for which the jacobian should be calculated
   * @warning The automatic differentiation is performed with the first call to calculateSparseJacobianForwardMap() or calculateSparseJacobianReverseMap(), s.t. these functions
   * should be called once before e.g. entering a time critical loop
   */
  SparseJacobianSymbolic(size_t function_input_dim, size_t function_output_dim, const CppAD::ADFun<Scalar>& function)
    : row_size_(function_output_dim)
    , col_size_(function_input_dim)
  {
    function_ = function;
    cppad_eigen_sparsity_pattern_handler_.init(function_input_dim, function_output_dim, function_);
    eigen_jac_vector_.resize(cppad_eigen_sparsity_pattern_handler_.getNumberOfNonZeros());
  };

  /**
   * @brief Returns a sparse Eigen map to access the jacobian.
   * @details The map needs just to be retrieved once.
   * @return sparse Eigen map to access the jacobian
   */
  ConstSparseJacobianMap<Scalar> getMap()
  {
    return ConstSparseJacobianMap<Scalar>(row_size_, col_size_, eigen_jac_vector_.size(), cppad_eigen_sparsity_pattern_handler_.getEigenRowIndexVec().data(),
                                          cppad_eigen_sparsity_pattern_handler_.getEigenColIndexVec().data(), eigen_jac_vector_.data());
  }

  /**
   * @brief Updates the entries of the jacobian based on the given input vector.
   * @details This function is an alias for updateJacobianForward() in order to provide a unified interface for all jacobian types.
   * @param input_vector input vector for updating the entries
   */
  void updateJacobian(const VectorXS<Scalar>& input_vector) { updateJacobianForward(input_vector); };

  /**
   * @brief Calculates the jacobian for the given input vector using CppADs forward mode.
   * @param input_vector input vector for which the jacobian should be calculated.
   */
  void updateJacobianForward(const VectorXS<Scalar>& input_vector)
  {
    function_.SparseJacobianForward(input_vector, cppad_eigen_sparsity_pattern_handler_.getSparsityPattern(), cppad_eigen_sparsity_pattern_handler_.getCppAdRowIndexVec(),
                                    cppad_eigen_sparsity_pattern_handler_.getCppAdColIndexVec(), eigen_jac_vector_, work_forward_);
  };

  /**
   * @brief Calculates the jacobian for the given input vector using CppADs reverse mode.
   * @param input_vector input vector for which the jacobian should be calculated.
   */
  void updateJacobianReverse(const VectorXS<Scalar>& input_vector)
  {
    function_.SparseJacobianReverse(input_vector, cppad_eigen_sparsity_pattern_handler_.getSparsityPattern(), cppad_eigen_sparsity_pattern_handler_.getCppAdRowIndexVec(),
                                    cppad_eigen_sparsity_pattern_handler_.getCppAdColIndexVec(), eigen_jac_vector_, work_reverse_);
  };

private:
  const size_t row_size_;
  const size_t col_size_;

  CppAD::ADFun<Scalar> function_;
  SparsityPatternHandler cppad_eigen_sparsity_pattern_handler_;

  CppAD::sparse_jacobian_work work_forward_;
  CppAD::sparse_jacobian_work work_reverse_;
  VectorXS<Scalar> eigen_jac_vector_;
};

// precompile for most common use case
extern template class SparseJacobianSymbolic<double>;
}  // namespace autojac
