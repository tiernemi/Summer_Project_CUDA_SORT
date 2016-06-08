/*
 * =====================================================================================
 *
 *       Filename:  flag.cpp
 *
 *    Description:  Source file for flag objects/
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

#include "../../inc/flag.hpp"

/* 
 * ===  MEMBER FUNCTION CLASS : Flag  ===================================================
 *         Name:  Flag
 *    Arguments:  const std::string & name - Name of argument.
 *                const Datatype & defaultVal - Default value of argument.
 *                const std::string & symbol - Symbol associated with argument.
 *  Description:  Initialises flag object.
 * ======================================================================================
 */

Flag::Flag(const std::string & name, const bool defaultVal, const std::string & symbol) 
	: CommandLineOption(name,symbol) {
	state = defaultVal ;
}		/* -----  end of member function Flag  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Flag  ==================================================
 *         Name:  processArgument
 *    Arguments:  const std::string & arg - Argument passed from command line.
 *  Description:  Set flag to true if detected in command line.
 * =====================================================================================
 */

void Flag::processOption(const std::string & arg) {
	state = true ;
}		/* -----  end of member function processArgument  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : Flag  =================================================
 *         Name: ~Flag
 *  Description: Destructor for flag.
 * =====================================================================================
 */

Flag::~Flag() {
	;
}		/* -----  end of member function   ----- */

