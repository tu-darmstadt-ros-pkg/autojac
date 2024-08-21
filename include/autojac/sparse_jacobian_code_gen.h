#pragma once

#include <cppad/cg.hpp>
#include <cppad/example/cppad_eigen.hpp>

#include "autojac/types.h"
#include "autojac/sparsity_pattern_handler.h"

namespace autojac
{
/**
 * @brief The SparseJacobianCodeGen class handles the calculation of a sparse jacobian of a function using code generation.
 * @tparam Scalar scalar type used for representing the jacobian (i.e. double or float).
 */
template <typename Scalar>
class SparseJacobianCodeGen
{
public:
  /**
   * @brief Alias for std::shared_ptr
   */
  using Ptr = typename std::shared_ptr<SparseJacobianCodeGen<Scalar>>;

  /**
   * @brief Alias for std::shared_ptr using a const reference to the object
   */
  using ConstPtr = typename std::shared_ptr<const SparseJacobianCodeGen<Scalar>>;

  /**
   * @brief Alias for CppAD::cg::CG of the Scalar type used to instantiate the class
   */
  using CGScalar = CppAD::cg::CG<Scalar>;

  /**
   * @brief Alias for CppAD::AD of the CGScalar type
   */
  using ADCGScalar = CppAD::AD<CGScalar>;

  /**
   * @brief Constructor that is used to generate a library baesd on an ADFun object.
   * @details The ADFun object can be created using the utility function generateCppAdFunction().
   * @param function_input_dim the size of the input vector of the function which is equal to the number of columns of the jacobian matrix
   * @param function_output_dim the size of the output vector of the function which is equal to the number of rows of the jacobian matrix
   * @param function reference to the function for which the jacobian should be generated
   * @param file_path path to the dynaimc library that should be created without the file extension (e.g. .so)
   */
  SparseJacobianCodeGen(size_t function_input_dim, size_t function_output_dim, CppAD::ADFun<CGScalar>& function, const std::string& file_path)
    : row_size_(function_output_dim)
    , col_size_(function_input_dim)

  {
    // init sparsity pattern handler
    cppad_eigen_sparsity_pattern_handler_.init(function_input_dim, function_output_dim, function);
    jac_vector_.resize(cppad_eigen_sparsity_pattern_handler_.getNumberOfNonZeros());

    // set up code generation
    CppAD::cg::ModelCSourceGen<Scalar> source_gen_model(function, "model");
    source_gen_model.setCreateSparseJacobian(true);
    CppAD::cg::ModelLibraryCSourceGen<Scalar> library_gen_model(source_gen_model);
    CppAD::cg::DynamicModelLibraryProcessor<Scalar> library_processor(library_gen_model, file_path);
    CppAD::cg::ClangCompiler<Scalar> compiler;

    // generate code and load library
    library_ptr_ = library_processor.createDynamicLibrary(compiler, true);
    model_ptr_ = library_ptr_->model("model");
  }

  /**
   * @brief Constructor that is used to load an existing library that has been created before.
   * @details This avoids the overhead of auto differentiating a CppAD function every time.
   * @param function_input_dim the size of the input vector of the function which is equal to the number of columns of the jacobian matrix
   * @param function_output_dim the size of the output vector of the function which is equal to the number of rows of the jacobian matrix
   * @param file_path path to the dynaimc library that should be created without the file extension (e.g. .so)
   */
  SparseJacobianCodeGen(size_t function_input_dim, size_t function_output_dim, const std::string& file_path)
    : row_size_(function_output_dim)
    , col_size_(function_input_dim)
  {
    // load library
    library_ptr_ = std::make_unique<CppAD::cg::LinuxDynamicLib<Scalar>>(file_path + CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION);
    model_ptr_ = library_ptr_->model("model");

    // init sparsity pattern handler
    cppad_eigen_sparsity_pattern_handler_.init(function_input_dim, function_output_dim, model_ptr_->JacobianSparsitySet());
    jac_vector_.resize(cppad_eigen_sparsity_pattern_handler_.getNumberOfNonZeros());
  }

  /**
   * @brief Returns a sparse Eigen map to access the jacobian.
   * @details The map needs just to be retrieved once.
   * @return sparse Eigen map to access the jacobian
   */
  ConstSparseJacobianMap<Scalar> getMap()
  {
    return ConstSparseJacobianMap<Scalar>(row_size_, col_size_, cppad_eigen_sparsity_pattern_handler_.getNumberOfNonZeros(),
                                          cppad_eigen_sparsity_pattern_handler_.getEigenRowIndexVec().data(), cppad_eigen_sparsity_pattern_handler_.getEigenColIndexVec().data(),
                                          jac_vector_.data());
  }

  /**
   * @brief Updates the entries of the jacobian based on the given input vector
   * @param input_vector input vector for updating the entries
   */
  void updateJacobian(const VectorXS<Scalar>& input_vector)
  {
    const size_t* row_ptr = cppad_eigen_sparsity_pattern_handler_.getCppAdRowIndexVec().data();
    const size_t* col_ptr = cppad_eigen_sparsity_pattern_handler_.getCppAdColIndexVec().data();
    CppAD::cg::ArrayView<const Scalar> x(input_vector.data(), input_vector.size());
    CppAD::cg::ArrayView<Scalar> y(jac_vector_.data(), jac_vector_.size());
    model_ptr_->SparseJacobian(x, y, &row_ptr, &col_ptr);
  }

private:
  const int row_size_;
  const int col_size_;

  std::unique_ptr<CppAD::cg::DynamicLib<Scalar>> library_ptr_;
  std::unique_ptr<CppAD::cg::GenericModel<Scalar>> model_ptr_;

  VectorXS<Scalar> jac_vector_;

  SparsityPatternHandler cppad_eigen_sparsity_pattern_handler_;
};

// precompile for most common use case
extern template class SparseJacobianCodeGen<double>;
}  // namespace autojac
