#include "autojac/function_generation_util.h"

namespace autojac
{
CppAD::ADFun<double> generateCppAdFunction(size_t function_input_dim, size_t function_output_dim,
                                           std::function<void(const VectorXS<CppAD::AD<double>>& cppad_x, VectorXS<CppAD::AD<double>>& cppad_y)> function);
CppAD::ADFun<CppAD::cg::CG<double>> generateCppAdFunction(size_t function_input_dim, size_t function_output_dim,
                                                          std::function<void(const Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1>& cppad_x,
                                                                             Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1>& cppad_y)>
                                                              function);

CppAD::ADFun<double> generateCppAdFunction(size_t function_input_dim, std::function<VectorXS<CppAD::AD<double>>(const VectorXS<CppAD::AD<double>>& cppad_x)> function);
CppAD::ADFun<CppAD::cg::CG<double>> generateCppAdFunction(
    size_t function_input_dim,
    std::function<Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1>(const Eigen::Matrix<CppAD::AD<CppAD::cg::CG<double>>, Eigen::Dynamic, 1>& cppad_x)> function);

}  // namespace autojac
