#pragma once

#include <iostream>
#include <cppad/utility/error_handler.hpp>

/** @file
 *  @brief This header file ensures that the CppAD errors do not stop the program.
 *  @details Especially in Debug mode the check of functions for NAN can stop the program if the function is not defined for a zero input vector.
 *  Yet, when evaluating the funtction it could be ensured by the caller, that the input vectors remain in a valid range.
 */

/**
 * @brief This function just prints an error message instead of throwing an exception
 * @param known
 * @param line
 * @param file
 * @param exp
 * @param msg
 */
void noThrowCallback(bool known, int line, const char* file, const char* exp, const char* msg);

extern CppAD::ErrorHandler autojac_error_handler;
