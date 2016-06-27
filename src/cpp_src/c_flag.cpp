/*
 * =====================================================================================
 *
 *       Filename:  c_flag.cpp
 *
 *    Description:  Source file for c_flag objects/
 *
 *        Version:  1.0
 *        Created:  08/06/16 15:28:19
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "../../inc/cpp_inc/c_flag.hpp"

/* 
 * ===  MEMBER FUNCTION CLASS : CFlag  ===================================================
 *         Name:  CFlag
 *    Arguments:  const std::string & name - Name of argument.
 *                const Datatype & defaultVal - Default value of argument.
 *                const std::string & symbol - Symbol associated with argument.
 *  Description:  Initialises c_flag object.
 * ======================================================================================
 */

CFlag::CFlag(const std::string & name, const bool defaultVal, const std::string & symbol) 
	: CommandLineOption(name,symbol) {
	state = defaultVal ;
}		/* -----  end of member function CFlag  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : CFlag  ==================================================
 *         Name:  processArgument
 *    Arguments:  const std::string & arg - Argument passed from command line.
 *  Description:  Set c_flag to true if detected in command line.
 * =====================================================================================
 */

void CFlag::processOption(const std::string & arg) {
	state = true ;
}		/* -----  end of member function processArgument  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : CFlag  =================================================
 *         Name: ~CFlag
 *  Description: Destructor for c_flag.
 * =====================================================================================
 */

CFlag::~CFlag() {
	;
}		/* -----  end of member function   ----- */

