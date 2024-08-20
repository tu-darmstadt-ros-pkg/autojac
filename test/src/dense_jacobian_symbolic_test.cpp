#include "gtest/gtest.h"

#include "autojac/types.h"

#include "autojac/function_generation_util.h"
#include "autojac/dense_jacobian_symbolic.h"

namespace autojac
{
namespace test
{
// test function
template <typename Scalar>
void testFunction(const VectorXS<Scalar>& x, VectorXS<Scalar>& y)
{
  y[0] = x[0] + x[2];
  y[1] = x[1] - x[3];
  y[2] = x[4] * x[4];
  y[3] = x[2] / x[1];
}

TEST(DenseJacobianSymbolic, Example)
{
  // create function
  const size_t input_dim = 5;
  const size_t output_dim = 4;

  CppAD::ADFun<double> func = generateCppAdFunction<double>(input_dim, output_dim, &testFunction<CppAD::AD<double>>);

  DenseJacobianSymbolic<double> jac(input_dim, output_dim, func);

  // define input
  Eigen::VectorXd eigen_x(input_dim);
  eigen_x << 3.0, 2.0, 7.0, 3.0, -1.0;

  // define expected jacobian
  Eigen::Matrix<double, 4, 5> expected_jac;
  expected_jac << 1, 0, 1, 0, 0,  //
      0, 1, 0, -1, 0,             //
      0, 0, 0, 0, -2,             //
      0, -0.25 * 7.0, 0.5, 0, 0;

  ConstDenseJacobianMap<double> eigen_jac_map = jac.getMap();

  // update values
  jac.updateJacobian(eigen_x);

  // check dimensions
  ASSERT_EQ(eigen_jac_map.rows(), eigen_jac_map.rows());
  ASSERT_EQ(eigen_jac_map.cols(), eigen_jac_map.cols());

  // check values
  for (int i = 0; i < eigen_jac_map.rows(); i++)
  {
    for (int j = 0; j < eigen_jac_map.cols(); j++)
    {
      EXPECT_EQ(eigen_jac_map(i, j), expected_jac(i, j));
    }
  }
}
}  // namespace test
}  // namespace autojac
