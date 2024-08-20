#include "gtest/gtest.h"

#include "autojac/types.h"
#include "autojac/function_generation_util.h"
#include "autojac/sparse_jacobian_code_gen.h"

#include <chrono>
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

TEST(SparseJacobianCodeGen, GeneateFunction)
{
  using CGD = CppAD::cg::CG<double>;
  using ADCG = CppAD::AD<CGD>;

  const size_t input_dim = 5;
  const size_t output_dim = 4;
  // create target driectory if not existent
  std::string create_dir_command = "mkdir -p ";
  create_dir_command.append(SOURCE_DIR "/test/test_codegen_files/");
  system(create_dir_command.c_str());

  std::string file_path(SOURCE_DIR "/test/test_codegen_files/sparse_code_gen_test");

  CppAD::ADFun<CGD> input_function = generateCppAdFunction<CGD>(input_dim, output_dim, &testFunction<ADCG>);

  // define input
  Eigen::VectorXd input_data(input_dim);
  input_data << 3.0, 2.0, 7.0, 3.0, -1.0;

  // define expected jacobian
  Eigen::Matrix<double, 4, 5> expected_jac;
  expected_jac << 1, 0, 1, 0, 0,  //
      0, 1, 0, -1, 0,             //
      0, 0, 0, 0, -2,             //
      0, -0.25 * 7.0, 0.5, 0, 0;

  {
    // create code gen object
    SparseJacobianCodeGen<double> jac(input_dim, output_dim, input_function, file_path);
    // retrieve map
    ConstSparseJacobianMap<double> result_jac = jac.getMap();
    // update values
    jac.updateJacobian(input_data);

    // check dimensions
    ASSERT_EQ(result_jac.rows(), expected_jac.rows());
    ASSERT_EQ(result_jac.cols(), expected_jac.cols());

    // check values
    for (int i = 0; i < result_jac.rows(); i++)
    {
      for (int j = 0; j < result_jac.cols(); j++)
      {
        EXPECT_EQ(result_jac.coeff(i, j), expected_jac(i, j));
      }
    }
  }

  {
    // create code gen object
    SparseJacobianCodeGen<double> jac_loaded(input_dim, output_dim, file_path);
    // retrieve map
    ConstSparseJacobianMap<double> loaded_result_jac = jac_loaded.getMap();
    // update values
    jac_loaded.updateJacobian(input_data);

    // check dimensions
    ASSERT_EQ(loaded_result_jac.rows(), expected_jac.rows());
    ASSERT_EQ(loaded_result_jac.cols(), expected_jac.cols());

    // check values
    for (int i = 0; i < loaded_result_jac.rows(); i++)
    {
      for (int j = 0; j < loaded_result_jac.cols(); j++)
      {
        EXPECT_EQ(loaded_result_jac.coeff(i, j), expected_jac(i, j));
      }
    }
  }
}

TEST(SparseJacobianCodeGen, RuntimeTest)
{
  const size_t input_dim = 5;
  const size_t output_dim = 4;

  std::string file_path(SOURCE_DIR "/test/test_codegen_files/sparse_code_gen_test");

  SparseJacobianCodeGen<double> jac_loaded(input_dim, output_dim, file_path);

  Eigen::VectorXd eigen_x(input_dim);
  eigen_x << 2.0, -3.5, 1.6, -3.4, 5.2;

  std::chrono::time_point start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < 1000000; i++)
    jac_loaded.updateJacobian(eigen_x);
  std::chrono::time_point stop = std::chrono::high_resolution_clock::now();

  std::cout << "\n[ Info     ] Time needed for 1,000,000 evaluations: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << " ms\n";
}

}  // namespace test
}  // namespace autojac
