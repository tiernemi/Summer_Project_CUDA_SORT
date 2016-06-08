/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  Main function for gpu-sort / cpu-sort comparison
 *
 *        Version:  1.0
 *        Created:  07/06/16 16:33:20
 *       Revision:  none
 *       Compiler:  g++
 *
 *         Author:  Michael Tierney (MT), tiernemi@tcd.ie
 *
 * =====================================================================================
 */

#include <stdlib.h>
#include <iostream>
#include <unordered_map>

// Custom headers. //
#include "../../inc/test_funcs.hpp"
#include "../../inc/fileloader.hpp"
#include "../../inc/cmd_parser.hpp"
#include "../../inc/cmd_line_argument.hpp"
#include "../../inc/flag.hpp"

int main(int argc, char *argv[]) {

	// Command Line Options //
	std::vector<CommandLineOption*> options ;
	Flag verbose("verbose",false,"v") ;
	options.push_back(&verbose) ;
	Flag help("help",false,"h") ;
	options.push_back(&help) ;
	CommandLineArgument<std::string> filename("filename","","f:") ;
	options.push_back(&filename) ;

	// Process Command Line Options //
	CommandParser::processArgs(argc, argv, options) ;

	// Read in the data //
	std::vector<Triangle> triangles ;
	std::vector<Camera> cameras ;
	FileLoader::loadFile(triangles,cameras,filename.getValue()) ;

	return EXIT_SUCCESS ;
}


