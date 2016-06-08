#ifndef CMD_LINE_ARGUMENT_HPP_G3NB7Z2R
#define CMD_LINE_ARGUMENT_HPP_G3NB7Z2R

/*
 * =====================================================================================
 *
 *       Filename:  cmd_line_argument.hpp
 *
 *    Description:  Command line argument class.
 *
 *        Version:  1.0
 *        Created:  08/06/16 14:48:53
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <string>

/* 
 * ===  CLASS  =========================================================================
 *         Name:  CommandLineOption
 *       Fields:  
 *  Description:  
 * =====================================================================================
 */

class CommandLineOption {
 public:
	CommandLineOption(const std::string & name, const std::string & symbol) ;
	virtual ~CommandLineOption() = 0 ;
	virtual void processOption(const std::string & arg) = 0 ;
	const std::string & getName() const {return name ;} ;
	const std::string & getSymbol() const {return symbol ;} ;
 protected:
	const std::string name ;
	const std::string symbol ;
} ;		/* -----  end of class CommandLineOption  ----- */

#endif
