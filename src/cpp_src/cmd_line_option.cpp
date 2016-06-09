/*
 * =====================================================================================
 *
 *       Filename:  cmd_line_argument.cpp
 *
 *    Description:  Source file for command line argument.
 *
 *        Version:  1.0
 *        Created:  08/06/16 14:57:40
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

// Custom Headers //
#include "../../inc/cmd_line_option.hpp"

/* 
 * ===  MEMBER FUNCTION CLASS : cmd_line_argument  =====================================
 *         Name:  CommandLineOption
 *    Arguments:  const std::string & name - Name of argument.
 *                const std::string & symbol - Symbol associated with argument.
 *  Description:  Initialises command line argument object.
 * =====================================================================================
 */

CommandLineOption::CommandLineOption(const std::string & name, const std::string & symbol) : 
	name{name}, symbol{symbol} { ; } ;

/* 
 * ===  MEMBER FUNCTION CLASS : cmd_line_option  ======================================
 *         Name:  ~CommandLineOption
 *  Description:  Destructor of command line option.
 * =====================================================================================
 */

CommandLineOption::~CommandLineOption() {
	;
}

