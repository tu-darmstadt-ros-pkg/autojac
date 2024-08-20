#include "gtest/gtest.h"

#include "autojac/function_generation_util.h"
#include "autojac/sparse_jacobian_symbolic.h"

#include <chrono>
namespace autojac
{
namespace test
{
// test function
template <typename Scalar>
void testFunction1(const VectorXS<Scalar>& x, VectorXS<Scalar>& y)
{
  y[0] = x[0] + x[2];
  y[1] = x[1] - x[3];
  y[2] = x[4] * x[4];
  y[3] = x[2] / x[1];
}

TEST(CppADEigenSparseJacobianTest, FunctionGenerationDefinition1)
{
  const double result_tolerance = 0.0001;
  const size_t input_dim = 5;
  const size_t output_dim = 4;

  CppAD::ADFun<double> func = generateCppAdFunction<double>(input_dim, output_dim, &testFunction1<CppAD::AD<double>>);

  SparseJacobianSymbolic<double> jac(input_dim, output_dim, func);

  // retrieve map
  ConstSparseJacobianMap<double> eigen_jac_map = jac.getMap();

  // check sparsity of map
  ASSERT_EQ(eigen_jac_map.nonZeros(), 7);

  // define test input
  Eigen::VectorXd eigen_x(input_dim);
  eigen_x << 2.0, -3.5, 1.6, -3.4, 5.2;

  // update values using forward mode
  jac.updateJacobianForward(eigen_x);

  EXPECT_NEAR(eigen_jac_map.coeff(0, 0), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(0, 2), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 1), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 3), -1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(2, 4), 10.4, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 1), -0.130612, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 2), -0.285714, result_tolerance);

  // update values using reverse mode
  jac.updateJacobianReverse(eigen_x);

  EXPECT_NEAR(eigen_jac_map.coeff(0, 0), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(0, 2), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 1), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 3), -1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(2, 4), 10.4, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 1), -0.130612, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 2), -0.285714, result_tolerance);

  // define new input
  eigen_x << -4.0, 2.1, 0.2, 2.7, -3.26;

  // update values using forward mode
  jac.updateJacobianForward(eigen_x);

  EXPECT_NEAR(eigen_jac_map.coeff(0, 0), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(0, 2), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 1), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 3), -1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(2, 4), -6.52, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 1), -0.0453515, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 2), 0.47619, result_tolerance);

  // update values using reverse mode
  jac.updateJacobianReverse(eigen_x);

  EXPECT_NEAR(eigen_jac_map.coeff(0, 0), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(0, 2), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 1), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 3), -1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(2, 4), -6.52, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 1), -0.0453515, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 2), 0.47619, result_tolerance);
}

// test function
template <typename Scalar>
VectorXS<Scalar> testFunction2(const VectorXS<Scalar>& x)
{
  VectorXS<Scalar> y(4);
  y[0] = x[0] + x[2];
  y[1] = x[1] - x[3];
  y[2] = x[4] * x[4];
  y[3] = x[2] / x[1];

  return y;
}

TEST(CppADEigenSparseJacobianTest, FunctionGenerationDefinition2)
{
  const double result_tolerance = 0.0001;
  const size_t input_dim = 5;
  const size_t output_dim = 4;

  CppAD::ADFun<double> func = generateCppAdFunction<double>(input_dim, &testFunction2<CppAD::AD<double>>);

  SparseJacobianSymbolic<double> jac(input_dim, output_dim, func);

  ConstSparseJacobianMap<double> eigen_jac_map = jac.getMap();

  ASSERT_EQ(eigen_jac_map.nonZeros(), 7);

  Eigen::VectorXd eigen_x(input_dim);
  eigen_x << 2.0, -3.5, 1.6, -3.4, 5.2;

  jac.updateJacobianForward(eigen_x);

  EXPECT_NEAR(eigen_jac_map.coeff(0, 0), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(0, 2), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 1), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 3), -1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(2, 4), 10.4, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 1), -0.130612, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 2), -0.285714, result_tolerance);

  jac.updateJacobianReverse(eigen_x);

  EXPECT_NEAR(eigen_jac_map.coeff(0, 0), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(0, 2), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 1), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 3), -1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(2, 4), 10.4, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 1), -0.130612, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 2), -0.285714, result_tolerance);

  eigen_x.resize(input_dim);
  eigen_x << -4.0, 2.1, 0.2, 2.7, -3.26;

  // CppAD computes a vector instead of a matrix
  jac.updateJacobianForward(eigen_x);

  EXPECT_NEAR(eigen_jac_map.coeff(0, 0), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(0, 2), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 1), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 3), -1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(2, 4), -6.52, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 1), -0.0453515, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 2), 0.47619, result_tolerance);

  jac.updateJacobianReverse(eigen_x);

  EXPECT_NEAR(eigen_jac_map.coeff(0, 0), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(0, 2), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 1), 1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(1, 3), -1, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(2, 4), -6.52, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 1), -0.0453515, result_tolerance);
  EXPECT_NEAR(eigen_jac_map.coeff(3, 2), 0.47619, result_tolerance);
}

TEST(CppADEigenSparseJacobianTest, RuntimeTest)
{
  const size_t input_dim = 5;
  const size_t output_dim = 4;

  CppAD::ADFun<double> func = generateCppAdFunction<double>(input_dim, &testFunction2<CppAD::AD<double>>);
  SparseJacobianSymbolic<double> jac(input_dim, output_dim, func);

  Eigen::VectorXd eigen_x(input_dim);
  eigen_x << 2.0, -3.5, 1.6, -3.4, 5.2;

  std::chrono::time_point start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 1000000; i++)
    jac.updateJacobian(eigen_x);
  std::chrono::time_point stop = std::chrono::high_resolution_clock::now();

  std::cout << "\n[ Info     ] Time needed for 1,000,000 evaluations: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms\n";
}
}  // namespace test
}  // namespace autojac
