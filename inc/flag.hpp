#ifndef FLAG_HPP_RCPIXP4A
#define FLAG_HPP_RCPIXP4A

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

/* 
 * ===  CLASS  =========================================================================
 *         Name:  Flag
 *       Fields:  
 *  Description:  
 * =====================================================================================
 */

class Flag : public CommandLineOption {
 public:
	Flag(const std::string & name, const bool defaultVal, const std::string & symbol) ;
	virtual ~Flag() ;
	virtual void processOption(const std::string & arg) ;
	bool getState() const {return state ;} ;
	void setState(const bool & val) { state = val ; } ;
 private:
	bool state ;
} ;		/* -----  end of class Flag  ----- */

#endif /* end of include guard: FLAG_HPP_RCPIXP4A */
