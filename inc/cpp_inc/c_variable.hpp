
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
 *         Name:  CVariable
 *       Fields:  Datatype value - Value of command line variable. 
 *  Description:  Class template used to generate command line variables.
 * =====================================================================================
 */

template <typename DataType>
class CVariable : public CommandLineOption {
 public:
	CVariable(const std::string & name, const DataType & defaultVal, const std::string & symbol) ;
	virtual ~CVariable() ;
	virtual void processOption(const std::string & arg) ;
	DataType getValue() const {return value ;} ;
	void setValue(const DataType & val) { value = val ; } ;
 private:
	DataType value ;
} ;		/* -----  end of class CVariable  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : c_variable  ============================================
 *         Name:  CVariable
 *    Arguments:  const std::string & name - Name of variable.
 *                const DataType & defaultVal - Default value of variable.
 *                const std::string & symbol - Symbol used to represent variable.
 *  Description:  Constructor for variable class.
 * =====================================================================================
 */

template <typename DataType>
CVariable<DataType>::CVariable(const std::string & name, const DataType & defaultVal, const std::string & symbol) : CommandLineOption(name, symbol) {
	;
}

/* 
 * ===  MEMBER FUNCTION CLASS : c_variable  ===========================================
 *         Name:  processOption
 *    Arguments:  const std::string & arg - Argument string passed by parser.
 *  Description:  Virtual function used to assign value to variable based on the command
 *                string passed by parser.
 * =====================================================================================
 */

template <typename DataType>
void CVariable<DataType>::processOption(const std::string & arg) {
	std::stringstream str(arg) ;
	str >> value ;
}		/* -----  end of member function function  ----- */

/* 
 * ===  MEMBER FUNCTION CLASS : c_variable  ===========================================
 *         Name:  ~CVariable()
 *  Description:  Destructor for CVariable.
 * =====================================================================================
 */

template <typename DataType>
CVariable<DataType>::~CVariable() {
	;
}		/* -----  end of member function function  ----- */

#endif /* end of include guard: COMMAND_LINE_ARG_HPP_RCPIXP4A */
