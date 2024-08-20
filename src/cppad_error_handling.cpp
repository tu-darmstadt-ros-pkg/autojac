#include "autojac/cppad_error_handling.h"

void noThrowCallback(bool known, int line, const char* file, const char* exp, const char* msg)
{
  std::cout << "\nCppAD error occured. Known: " << known << " Line: " << line << " File: " << std::string(file) << " Expression: " << std::string(exp)
            << " Message: " << std::string(msg) << std::endl;
}

CppAD::ErrorHandler autojac_error_handler(noThrowCallback);