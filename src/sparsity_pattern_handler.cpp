#include "autojac/sparsity_pattern_handler.h"

void autojac::SparsityPatternHandler::init(int input_dim, int output_dim, const std::vector<std::set<size_t>>& sparsity)
{
  assert(sparsity.size() == output_dim);

  row_size_ = output_dim;
  col_size_ = input_dim;
  s_ = sparsity;

  nnz_ = countNonZeros(s_);
  setEigenIndexVectors(s_);
  setCppAdndexVectors(s_);
}

int autojac::SparsityPatternHandler::getNumberOfNonZeros() { return nnz_; }

const std::vector<int>& autojac::SparsityPatternHandler::getEigenRowIndexVec() { return eigen_row_outer_index_; }

const std::vector<int>& autojac::SparsityPatternHandler::getEigenColIndexVec() { return eigen_col_inner_index_; }

const std::vector<size_t>& autojac::SparsityPatternHandler::getCppAdRowIndexVec() { return cppad_row_index_; }

const std::vector<size_t>& autojac::SparsityPatternHandler::getCppAdColIndexVec() { return cppad_col_index_; }

const std::vector<std::set<size_t>>& autojac::SparsityPatternHandler::getSparsityPattern() { return s_; }

void autojac::SparsityPatternHandler::setEigenIndexVectors(const std::vector<std::set<size_t>>& s)
{
  eigen_row_outer_index_.resize(row_size_ + 1);
  eigen_col_inner_index_.reserve(nnz_);
  eigen_col_inner_index_.clear();

  eigen_row_outer_index_[0] = 0;

  for (int i = 0; i < s.size(); i++)
  {
    for (const size_t& col_idx : s[i])
    {
      eigen_col_inner_index_.push_back(col_idx);
    }
    eigen_row_outer_index_[i + 1] = eigen_row_outer_index_[i] + s[i].size();
  }
}

void autojac::SparsityPatternHandler::setCppAdndexVectors(const std::vector<std::set<size_t>>& s)
{
  cppad_col_index_.reserve(nnz_);
  cppad_col_index_.clear();
  cppad_row_index_.reserve(nnz_);
  cppad_row_index_.clear();

  for (int i = 0; i < s.size(); i++)
  {
    for (const size_t& col_idx : s[i])
    {
      cppad_row_index_.push_back(i);
      cppad_col_index_.push_back(col_idx);
    }
  }
}

size_t autojac::SparsityPatternHandler::countNonZeros(const std::vector<std::set<size_t>>& s)
{
  size_t nnz_counter = 0;
  for (int i = 0; i < s.size(); i++)
  {
    nnz_counter += s[i].size();
  }
  return nnz_counter;
}

CppAD::sparse_rc<std::vector<size_t>> autojac::SparsityPatternHandler::generateDiagonalPattern(size_t col_size)
{
  CppAD::sparse_rc<std::vector<size_t>> eye_mat_pattern(col_size, col_size, col_size);
  for (size_t i = 0; i < col_size; i++)
  {
    eye_mat_pattern.set(i, i, i);
  }
  return eye_mat_pattern;
}
