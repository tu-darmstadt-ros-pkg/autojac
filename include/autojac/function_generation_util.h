#pragma once

#include <cppad/cg.hpp>
#include <cppad/example/cppad_eigen.hpp>

#include "autojac/types.h"

/**
 * @file
 * @brief Header file containing utility functions for generating CppAD::ADFun objects used by the classes for generating the jacobian.
 */

namespace autojac
{
/**
 * @brief Generates a CppAD function based on a function handle.
 * @details Function handle type void f(const VectorXS<CppAD::AD<Scalar>>& cppad_x, VectorXS<CppAD::AD<Scalar>>& cppad_y)
 * @param function_input_dim size of the input vector
 * @param function_output_dim size of the output vector
 * @param function function handle
 * @return CppAD Function
 */
template <typename Scalar>
CppAD::ADFun<Scalar> generateCppAdFunction(size_t function_input_dim, size_t function_output_dim,
                                           std::function<void(const VectorXS<CppAD::AD<Scalar>>& cppad_x, VectorXS<CppAD::AD<Scalar>>& cppad_y)> function)
{
  VectorXS<CppAD::AD<Scalar>> cppad_x(function_input_dim);
  VectorXS<CppAD::AD<Scalar>> cppad_y(function_output_dim);

  CppAD::Independent(cppad_x);
  function(cppad_x, cppad_y);

  return CppAD::ADFun<Scalar>(cppad_x, cppad_y);
}

/**
 * @brief Generates a CppAD function based on a function handle.
 * @details Function handle type VectorXS<CppAD::AD<Scalar>>(const VectorXS<CppAD::AD<Scalar>>& cppad_x)
 * @param function_input_dim size of the input vector
 * @param function function handle
 * @return CppAD Function
 */
template <typename Scalar>
CppAD::ADFun<Scalar> generateCppAdFunction(
    size_t function_input_dim, std::function<Eigen::Matrix<CppAD::AD<Scalar>, Eigen::Dynamic, 1>(const Eigen::Matrix<CppAD::AD<Scalar>, Eigen::Dynamic, 1>& cppad_x)> function)
{
  VectorXS<CppAD::AD<Scalar>> cppad_x(function_input_dim);
  CppAD::Independent(cppad_x);

  VectorXS<CppAD::AD<Scalar>> cppad_y = function(cppad_x);

  return CppAD::ADFun<Scalar>(cppad_x, cppad_y);
}

extern CppAD::ADFun<double> generateCppAdFunction(size_t function_input_dim, size_t function_output_dim,
                                                  std::function<void(const VectorXS<CppAD::AD<double>>& cppad_x, VectorXS<CppAD::AD<double>>& cppad_y)> function);
extern CppAD::ADFun<CppAD::cg::CG<double>> generateCppAdFunction(size_t function_input_dim, size_t function_output_dim,
                                                                 std::function<void(const Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1>& cppad_x,
                                                                                    Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1>& cppad_y)>
                                                                     function);

extern CppAD::ADFun<double> generateCppAdFunction(size_t function_input_dim, std::function<VectorXS<CppAD::AD<double>>(const VectorXS<CppAD::AD<double>>& cppad_x)> function);
extern CppAD::ADFun<CppAD::cg::CG<double>> generateCppAdFunction(
    size_t function_input_dim,
    std::function<Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1>(const Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1>& cppad_x)> function);

}  // namespace autojac
