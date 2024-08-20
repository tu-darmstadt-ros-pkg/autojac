# autojac
This package contains classes and functions for using the automatic differentiation capabilities of [CppAD](https://github.com/coin-or/CppAD) to calculate the jacobian of a function.
The input and output data of the function and the jacobian are represented by [Eigen](https://eigen.tuxfamily.org/) matrix types.
For representing the jacobian dense and sparse variants can be used.
Furthermore, code generation using [CppADCodeGen](https://github.com/joaoleal/CppADCodeGen) for efficient calculation of the jacobian can be utilized.

## Documentation
When building the package you can use the flag '-DBUILD_DOC=TRUE' to build the documentation. You can access it in the doc folder afterwards.

### Example
The following examples show how to generate and evaluate a sparse jacobian of a function using generated code.
The code for symbolic evaluation is very similar. The main difference is that you do not have to specify a file path there and that you can not load a previously generated jacobian.

The interface for dense jacobians works just like for sparse jacobians.
```c++
#include "autojac/function_generation_util.h"
#include "autojac/sparse_jacobian_code_gen.h"

// define a template vector function for which the jacobian should be generated
template <typename Scalar>
void exampleFunction(const VectorXS<Scalar>& x, VectorXS<Scalar>& y)
{
  y[0] = x[0] + x[2];
  y[1] = x[1] - x[3];
  y[2] = x[4] * x[4];
  y[3] = x[2] / x[1];
}

// example function for generating the jacobian
void example()
{
  // define aliases to shorten code
  using CGD = CppAD::cg::CG<double>;
  using ADCG = CppAD::AD<CGD>;

  // define input and output size of the function
  const size_t input_dim = 5;
  const size_t output_dim = 4;
  
  // create a CppAD function object based on the exampleFunction
  CppAD::ADFun<CGD> input_function = generateCppAdFunction<CGD>(input_dim, output_dim, &exampleFunction<ADCG>);
  
  // define path to the library for code gerneration without the file extension
  const std::string library_path = "path/to/library";

  // create object for code generation and holding the data of the jacobian
  SparseJacobianCodeGen<double> jac(input_dim, output_dim, input_function, file_path);

  // retrieve Eigen map that can be used for read-only access to the jacobian
  ConstSparseJacobianMap<double> result_jac = jac.getMap();
  
  // define input data
  Eigen::VectorXd input_data(input_dim);
  
  // update values of the jacobian for the given input vector
  jac.updateJacobian(input_data);
  
  // use values of the jacobian e.g. for printing the result
  std::cout << result_jac;
}
```
Note that you can store the map before the values are updated and that the map keeps valid after each update.

Once the code for evaluating the jacobian is created, you can avoid the overhead of creating the library and load it directly as shown in this example:
```c++
#include "autojac/sparse_jacobian_code_gen.h"

// example function for loading the jacobian
void example()
{
  // define path to the library containing the previously gernerated code without the file extension
  const std::string library_path = "path/to/library";

  // create object for loading the library and holding the data of the jacobian
  SparseJacobianCodeGen<double> jac(input_dim, output_dim, file_path);

  // retrieve Eigen map that can be used for read-only access to the jacobian
  ConstSparseJacobianMap<double> result_jac = jac.getMap();
  
  // define input data
  Eigen::VectorXd input_data(input_dim);
  
  // update values of the jacobian for the given input vector
  jac.updateJacobian(input_data);
  
  // use values of the jacobian e.g. for printing the result
  std::cout << result_jac;
}
```
## Installation
While the package is set up to be build using [ament](https://design.ros2.org/articles/ament.html), it has no ROS dependencies.
To clone and build the packages of CppAD and CppADCodeGen you can execute the install script from the top level folder.
```
$ ./scripts/install_cppad_and_cppadcg.sh
```
After this you can build the package using `colcon build`.
For using code generation the clang compiler needs to be installed. This can be done using
```
$ sudo apt install clang
```
This package has been developed originally using catkin as build system. You can find this version on the "catkin_version" branch.

## Future Development & Contribution
The project during which the package was developed has been discontinued.
But in case you find bugs, typos or have suggestions for improvements feel free to open an issue.
We would especially appreciate Pull Requests fixing open issues.

## Authors
- Felix Biemüller
- Julian Rau
