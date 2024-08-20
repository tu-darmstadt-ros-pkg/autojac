#pragma once

#include <cppad/example/cppad_eigen.hpp>

namespace autojac
{
/**
 * @brief The CppADEigenSparsityPatternHandler class handles indexing and the sparsity pattern.
 * @tparam Scalar a scalar type.
 */
class CppADEigenSparsityPatternHandler
{
public:
  /**
   * @brief CppADEigenSparsityPatternHandler constructor.
   */
  CppADEigenSparsityPatternHandler() = default;

  template <typename Scalar>
  /**
   * @brief Initializes the sparsity pattern vectors given a CppAD function
   * @param input_dim number of input variables (number of columns of the jacobian matrix).
   * @param output_dim number of output variables of the function (number of rows of the jacobian matrix).
   * @param func the function which jacobian should be calculated.
   */
  void init(int input_dim, int output_dim, CppAD::ADFun<Scalar>& func)
  {
    row_size_ = output_dim;
    col_size_ = input_dim;

    std::vector<std::set<size_t>> r(input_dim);
    for (int i = 0; i < input_dim; i++)
    {
      r[i].insert(i);
    }
    init(input_dim, output_dim, func.ForSparseJac(col_size_, r));
  }

  /**
   * @brief Initializes the sparsity pattern vectors given a CppAD sparsity pattern
   * @param input_dim number of input variables (number of columns of the jacobian matrix).
   * @param output_dim number of output variables of the function (number of rows of the jacobian matrix).
   * @param sparsity sparsity pattern given as vector of sets
   */
  void init(int input_dim, int output_dim, const std::vector<std::set<size_t>>& sparsity);

  /**
   * @brief getNonZeros get number of non-zero values.
   * @return number of non-zero values.
   */
  int getNumberOfNonZeros();

  /**
   * @brief Returns a reference to a vector containing the row indices for an Eigen sparsity pattern
   * @return row indices for an Eigen sparsity pattern
   */
  const std::vector<int>& getEigenRowIndexVec();

  /**
   * @brief Returns a reference to a vector containing the column indices for an Eigen sparsity pattern
   * @return column indices for an Eigen sparsity pattern
   */
  const std::vector<int>& getEigenColIndexVec();

  /**
   * @brief Returns a reference to a vector containing the row indices for a CppAd sparsity pattern
   * @return row indices for a CppAd sparsity pattern
   */
  const std::vector<size_t>& getCppAdRowIndexVec();

  /**
   * @brief Returns a reference to a vector containing the column indices for a CppAd sparsity pattern
   * @return column` indices for a CppAd sparsity pattern
   */
  const std::vector<size_t>& getCppAdColIndexVec();

  /**
   * @brief Returns the sparsity pattern of the matrix as vector of sets.
   * @return The sparsity pattern of the matrix
   */
  const std::vector<std::set<size_t>>& getSparsityPattern();

private:
  void setEigenIndexVectors(const std::vector<std::set<size_t>>& s);
  void setCppAdndexVectors(const std::vector<std::set<size_t>>& s);

  static size_t countNonZeros(const std::vector<std::set<size_t>>& s);
  static CppAD::sparse_rc<std::vector<size_t>> generateDiagonalPattern(size_t col_size);

  int row_size_;
  int col_size_;
  int nnz_;

  std::vector<std::set<size_t>> s_;

  std::vector<std::set<size_t>> cppad_pattern_;

  std::vector<size_t> cppad_row_index_;
  std::vector<size_t> cppad_col_index_;

  std::vector<int> eigen_row_outer_index_;
  std::vector<int> eigen_col_inner_index_;
};

}  // namespace autojac
