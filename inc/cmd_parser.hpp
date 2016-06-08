#ifndef COMMAND_PARSER_HPP_QRU3HNPE
#define COMMAND_PARSER_HPP_QRU3HNPE

/*
 * =====================================================================================
 *
 *       Filename:  cmd_parser.hpp
 *
 *    Description:  Helper static class for parsing command line arguments.
 *
 *        Version:  1.0
 *        Created:  08/06/16 15:57:48
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <vector>
#include "cmd_line_option.hpp"
/* 
 * ===  CLASS  =========================================================================
 *         Name:  CommandParser
 *       Fields:  
 *  Description:  
 * =====================================================================================
 */

class CommandParser {
 public:
	 static void processArgs(int argc, char * argv[], std::vector<CommandLineOption*> & options) ;
} ;		/* -----  end of class CommandParser  ----- */

#endif /* end of include guard: COMMAND_PARSER_HPP_QRU3HNPE */
