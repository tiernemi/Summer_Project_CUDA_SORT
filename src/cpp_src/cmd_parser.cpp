/*
 * =====================================================================================
 *
 *       Filename:  cmd_parser.cpp
 *
 *    Description:  Source file for command parsing class.
 *
 *        Version:  1.0
 *        Created:  08/06/16 16:15:40
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <string>
#include <iostream>
#include "stdlib.h"
#include "getopt.h"

// Custom Headers //
#include "../../inc/cmd_parser.hpp"
#include "../../inc/cmd_line_option.hpp"

/* 
 * ===  MEMBER FUNCTION CLASS : CommandParser  ======================================
 *         Name:  processArgs
 *    Arguments:  int argc - Number of cmd line arguments.
 *                char * argv[] - Array containg arguments.
 *                std::vector<CommandLineOption*> Array containing program options.
 *  Description:  Parses command line options based on the passed options array.
 * =====================================================================================
 */

void CommandParser::processArgs(int argc, char * argv[], std::vector<CommandLineOption*> & options) {
	std::string optstr("") ;
	for (unsigned int i = 0 ; i < options.size() ; ++i) {
		optstr += options[i]->getSymbol() ;
	}
	int choice;
	while (1) {
		bool noFound = true ;
		choice = getopt(argc, argv, optstr.c_str());	
		if (choice == -1)
			break;
		for (unsigned int j = 0 ; j < options.size() ; ++j) {
			char symbol = char(choice) ;
			std::string symbolString = std::string(1,symbol) ;
			if (options[j]->getSymbol().find(symbolString) != std::string::npos) {
				std::string args ;
				if (optarg == NULL) {
					args = "" ;
				} else {
					args = optarg ;
				}
				options[j]->processOption(args) ;
				noFound = false ;
				break ;
			}
		}
		if (noFound) {
			std::cerr << "Unknown option, Type -h for help" << std::endl ;
			exit(-2) ;
		}
	}
}		/* -----  end of member function processArgs  ----- */
