
#ifndef COMMAND_LINE_ARG_HPP_RCPIXP4A
#define COMMAND_LINE_ARG_HPP_RCPIXP4A

/*
 * =====================================================================================
 *
 *       Filename:  flag.hpp
 *
 *    Description:  Header file for flag class.
 *
 *        Version:  1.0
 *        Created:  08/06/16 15:25:21
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include "cmd_line_option.hpp"
#include <sstream>

/* 
 * ===  CLASS  =========================================================================
 *         Name:  CommandLineArgument
 *       Fields:  
 *  Description:  
 * =====================================================================================
 */

template <typename DataType>
class CommandLineArgument : public CommandLineOption {
 public:
	CommandLineArgument(const std::string & name, const DataType & defaultVal, const std::string & symbol) ;
	virtual ~CommandLineArgument() ;
	virtual void processOption(const std::string & arg) ;
	DataType getValue() const {return value ;} ;
	void setValue(const DataType & val) { value = val ; } ;
 private:
	DataType value ;
} ;		/* -----  end of class CommandLineArgument  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : cmd_line_argument  ======================================
 *         Name:  function
 *    Arguments:  
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

template <typename DataType>
CommandLineArgument<DataType>::CommandLineArgument(const std::string & name, const DataType & defaultVal, const std::string & symbol) : CommandLineOption(name, symbol) {
	;
}

/* 
 * ===  MEMBER FUNCTION CLASS : cmd_line_argument  ======================================
 *         Name:  function
 *    Arguments:  
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

template <typename DataType>
void CommandLineArgument<DataType>::processOption(const std::string & arg) {
	std::stringstream str(arg) ;
	str >> value ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : cmd_line_argument  ======================================
 *         Name:  function
 *    Arguments:  
 *      Returns:  
 *  Description:  
 * =====================================================================================
 */

template <typename DataType>
CommandLineArgument<DataType>::~CommandLineArgument() {
	;
}		/* -----  end of member function function  ----- */

#endif /* end of include guard: COMMAND_LINE_ARG_HPP_RCPIXP4A */
