#include "ConvertCommand.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/shared_ptr.hpp>
#include "Ranally/Operation/XmlParser.h"
#include "Ranally/Operation/Operation-xml.h"
#include "Ranally/Language/AlgebraParser.h"
#include "Ranally/Language/AnnotateVisitor.h"
#include "Ranally/Language/AstDotVisitor.h"
#include "Ranally/Language/FlowgraphDotVisitor.h"
#include "Ranally/Language/IdentifyVisitor.h"
#include "Ranally/Language/ScriptVertex.h"
#include "Ranally/Language/ThreadVisitor.h"
#include "Ranally/Language/XmlParser.h"



namespace ranally {
namespace {

void showConvertHelp()
{
  std::cout <<
    "usage: ranally convert [--help] LANGUAGE [ARGS]\n"
    "\n"
    "Convert the script to a target language.\n"
    "\n"
    "languages:\n"
    "  ranally             Round-trip script\n"
    "  dot                 Convert script to Dot graph\n"
    "  c++                 Convert script to C++ code\n"
    "  python              Convert script to C++ code for Python extension\n"
    "  xml                 Convert script to XML\n"
    "\n"
    "See 'ranally convert LANGUAGE --help' for more information on a specific\n"
    "language.\n"
    ;
}



void showConvertDotHelp()
{
  std::cout <<
    "usage: ranally convert dot [--help] GRAPH_TYPE [ARGS]\n"
    "\n"
    "Convert the script to a dot graph.\n"
    "\n"
    "graph types:\n"
    "  ast                 Abstract syntax tree\n"
    "  flowgraph           Flowgraph\n"
    "\n"
    "See 'ranally convert dot GRAPH_TYPE --help' for more information on a\n"
    "specific graph type.\n"
    ;
}



void showConvertDotAstHelp()
{
  std::cout <<
    "usage: ranally convert dot ast [--help] [--with-cfg] [--with-use]\n"
    "                               INPUT_SCRIPT OUTPUT_SCRIPT\n"
    "\n"
    "Convert the script to a dot graph containing the abstract syntax tree.\n"
    "\n"
    "  INPUT_SCRIPT        Script to convert or - to read from standard input\n"
    "  OUTPUT_SCRIPT       File to write result to\n"
    "\n"
    "The result is written to standard output if no output script is provided\n"
    ;
}



void showConvertDotFlowgraphHelp()
{
  std::cout <<
    "usage: ranally convert dot flowgraph [--help] INPUT_SCRIPT OUTPUT_SCRIPT\n"
    "\n"
    "Convert the script to a dot graph containing the flow graph.\n"
    "\n"
    "  INPUT_SCRIPT        Script to convert or - to read from standard input\n"
    "  OUTPUT_SCRIPT       File to write result to\n"
    "\n"
    "The result is written to standard output if no output script is provided\n"
    ;
}



// void showConvertRanallyHelp()
// {
//   std::cout <<
//     "usage: ranally convert ranally INPUT_SCRIPT [OUTPUT_SCRIPT]\n"
//     "\n"
//     "Convert the script to a ranally script (round-trip).\n"
//     "\n"
//     "  INPUT_SCRIPT        Script to convert or - to read from standard input\n"
//     "  OUTPUT_SCRIPT       File to write result to\n"
//     "\n"
//     "The result is written to standard output if no output script is provided\n"
//     ;
// }



void showConvertXmlHelp()
{
  std::cout <<
    "usage: ranally convert xml [--help] INPUT_SCRIPT OUTPUT_SCRIPT\n"
    "\n"
    "Convert the script to xml.\n"
    "\n"
    "  INPUT_SCRIPT        Script to convert or - to read from standard input\n"
    "  OUTPUT_SCRIPT       File to write result to\n"
    "\n"
    "The result is written to standard output if no output script is provided\n"
    ;
}

} // Anonymous namespace



ConvertCommand::ConvertCommand(
  int argc,
  char** argv)

  : Command(argc, argv)

{
}



ConvertCommand::~ConvertCommand()
{
}



int ConvertCommand::convertToRanally(
  int /* argc */,
  char** /* argv */)
{
  std::cout << "Conversion to Ranally script not supported yet\n";
  return EXIT_SUCCESS;
}



int ConvertCommand::convertToCpp(
  int /* argc */,
  char** /* argv */)
{
  std::cout << "Conversion to C++ not supported yet\n";
  return EXIT_SUCCESS;
}



UnicodeString ConvertCommand::convertToDotAst(
  UnicodeString const& xml,
  int modes)
{
  boost::shared_ptr<ranally::language::ScriptVertex> tree(
    ranally::language::XmlParser().parse(xml));

  ranally::language::ThreadVisitor threadVisitor;
  tree->Accept(threadVisitor);

  ranally::language::IdentifyVisitor identifyVisitor;
  tree->Accept(identifyVisitor);

  ranally::operation::OperationsPtr operations(
    ranally::operation::XmlParser().parse(ranally::operation::operationsXml));
  ranally::language::AnnotateVisitor annotateVisitor(operations);
  tree->Accept(annotateVisitor);

  ranally::AstDotVisitor astDotVisitor(modes);
  tree->Accept(astDotVisitor);

  return astDotVisitor.script();
}



int ConvertCommand::convertToDotAst(
  int argc,
  char** argv)
{
  int status = EXIT_FAILURE;

  if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
    // No arguments, or the help option.
    showConvertDotAstHelp();
    status = EXIT_SUCCESS;
  }
  else {
    int currentArgumentId = 1;
    int modes = 0x0;
    while(currentArgumentId < argc) {
      if(std::strcmp(argv[currentArgumentId], "--with-cfg") == 0) {
        modes |= ranally::AstDotVisitor::ConnectingCfg;
        ++currentArgumentId;
      }
      else if(std::strcmp(argv[currentArgumentId], "--with-use") == 0) {
        modes |= ranally::AstDotVisitor::ConnectingUses;
        ++currentArgumentId;
      }
      else {
        break;
      }
    }

    if(currentArgumentId == argc) {
      std::cerr << "Not enough arguments.\n";
      showConvertDotAstHelp();
      status = EXIT_FAILURE;
    }
    else if(argc - currentArgumentId > 3) {
      std::cerr << "Too many arguments.\n";
      showConvertDotAstHelp();
      status = EXIT_FAILURE;
    }
    else {
      std::string inputFileName =
        std::strcmp(argv[currentArgumentId], "-") != 0
          ? argv[currentArgumentId] : "";
      ++currentArgumentId;
      std::string outputFileName = currentArgumentId == argc - 1
        ? argv[currentArgumentId] : "";
      UnicodeString xml;
      ranally::language::AlgebraParser parser;

      if(inputFileName.empty()) {
        // Read script from the standard input stream.
        std::ostringstream script;
        script << std::cin.rdbuf();
        xml = parser.parseString(UnicodeString(script.str().c_str()));
      }
      else {
        // Read script from a file.
        xml = parser.parseFile(UnicodeString(inputFileName.c_str()));
      }

      std::string dotScript = ranally::util::encodeInUTF8(
        convertToDotAst(xml, modes));

      if(outputFileName.empty()) {
        std::cout << dotScript;
      }
      else {
        std::ofstream file(outputFileName.c_str());
        file << dotScript;
      }

      status = EXIT_SUCCESS;
    }
  }

  return status;
}



UnicodeString ConvertCommand::convertToDotFlowgraph(
  UnicodeString const& xml)
{
  boost::shared_ptr<ranally::language::ScriptVertex> tree(
    ranally::language::XmlParser().parse(xml));

  ranally::language::ThreadVisitor threadVisitor;
  tree->Accept(threadVisitor);

  ranally::language::IdentifyVisitor identifyVisitor;
  tree->Accept(identifyVisitor);

  ranally::operation::OperationsPtr operations(
    ranally::operation::XmlParser().parse(ranally::operation::operationsXml));
  ranally::language::AnnotateVisitor annotateVisitor(operations);
  tree->Accept(annotateVisitor);

  ranally::FlowgraphDotVisitor flowgraphDotVisitor;
  tree->Accept(flowgraphDotVisitor);

  return flowgraphDotVisitor.script();
}



int ConvertCommand::convertToDotFlowgraph(
  int argc,
  char** argv)
{
  int status = EXIT_FAILURE;

  if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
    // No arguments, or the help option.
    showConvertDotFlowgraphHelp();
    status = EXIT_SUCCESS;
  }
  else {
    int currentArgumentId = 1;

    if(currentArgumentId == argc) {
      std::cerr << "Not enough arguments.\n";
      showConvertDotAstHelp();
      status = EXIT_FAILURE;
    }
    else if(argc - currentArgumentId > 3) {
      std::cerr << "Too many arguments.\n";
      showConvertDotAstHelp();
      status = EXIT_FAILURE;
    }
    else {
      std::string inputFileName =
        std::strcmp(argv[currentArgumentId], "-") != 0
          ? argv[currentArgumentId] : "";
      ++currentArgumentId;
      std::string outputFileName = currentArgumentId == argc - 1
        ? argv[currentArgumentId] : "";
      UnicodeString xml;
      ranally::language::AlgebraParser parser;

      if(inputFileName.empty()) {
        // Read script from the standard input stream.
        std::ostringstream script;
        script << std::cin.rdbuf();
        xml = parser.parseString(UnicodeString(script.str().c_str()));
      }
      else {
        // Read script from a file.
        xml = parser.parseFile(UnicodeString(inputFileName.c_str()));
      }

      std::string dotScript = ranally::util::encodeInUTF8(
        convertToDotFlowgraph(xml));

      if(outputFileName.empty()) {
        std::cout << dotScript;
      }
      else {
        std::ofstream file(outputFileName.c_str());
        file << dotScript;
      }

      status = EXIT_SUCCESS;
    }
  }

  return status;
}



int ConvertCommand::convertToDot(
  int argc,
  char** argv)
{
  int status = EXIT_FAILURE;

  if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
    // No arguments, or the help option.
    showConvertDotHelp();
    status = EXIT_SUCCESS;
  }
  else if(std::strcmp(argv[1], "ast") == 0) {
    status = convertToDotAst(argc - 1, argv + 1);
  }
  else if(std::strcmp(argv[1], "flowgraph") == 0) {
    status = convertToDotFlowgraph(argc - 1, argv + 1);
  }
  else {
    std::cerr << "Unknown graph type: " << argv[1] << "\n";
    std::cerr << "See 'ranally convert dot --help' for list of types.\n";
    status = EXIT_FAILURE;
  }

  return status;
}



int ConvertCommand::convertToPython(
  int /* argc */,
  char** /* argv */)
{
  std::cout << "Conversion to Python not supported yet\n";
  return EXIT_SUCCESS;
}



int ConvertCommand::convertToXml(
  int argc,
  char** argv)
{
  int status = EXIT_FAILURE;

  if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
    // No arguments, or the help option.
    showConvertXmlHelp();
    status = EXIT_SUCCESS;
  }
  else {
    int currentArgumentId = 1;

    if(argc - currentArgumentId > 2) {
      std::cerr << "Too many arguments.\n";
      showConvertXmlHelp();
      status = EXIT_FAILURE;
    }
    else {
      std::string inputFileName =
        std::strcmp(argv[currentArgumentId], "-") != 0
          ? argv[currentArgumentId] : "";
      ++currentArgumentId;
      std::string outputFileName = currentArgumentId == argc - 1
        ? argv[currentArgumentId] : "";
      UnicodeString xml;
      ranally::language::AlgebraParser parser;

      if(inputFileName.empty()) {
        // Read script from the standard input stream.
        std::ostringstream script;
        script << std::cin.rdbuf();
        xml = parser.parseString(UnicodeString(script.str().c_str()));
      }
      else {
        // Read script from a file.
        xml = parser.parseFile(UnicodeString(inputFileName.c_str()));
      }

      if(outputFileName.empty()) {
        std::cout << ranally::util::encodeInUTF8(xml);
      }
      else {
        std::ofstream file(outputFileName.c_str());
        file << ranally::util::encodeInUTF8(xml);
      }

      status = EXIT_SUCCESS;
    }
  }

  return status;
}



int ConvertCommand::execute()
{
  int status = EXIT_FAILURE;

  if(argc() == 1 || std::strcmp(argv()[1], "--help") == 0) {
    // No arguments, or the help option.
    showConvertHelp();
    status = EXIT_SUCCESS;
  }
  else if(std::strcmp(argv()[1], "ranally") == 0) {
    status = convertToRanally(argc() - 1, argv() + 1);
  }
  else if(std::strcmp(argv()[1], "dot") == 0) {
    status = convertToDot(argc() - 1, argv() + 1);
  }
  else if(std::strcmp(argv()[1], "c++") == 0) {
    status = convertToCpp(argc() - 1, argv() + 1);
  }
  else if(std::strcmp(argv()[1], "python") == 0) {
    status = convertToPython(argc() - 1, argv() + 1);
  }
  else if(std::strcmp(argv()[1], "xml") == 0) {
    status = convertToXml(argc() - 1, argv() + 1);
  }
  else {
    std::cerr << "Unknown target language: " << argv()[1] << "\n";
    std::cerr << "See 'ranally convert --help' for list of languages.\n";
    status = EXIT_FAILURE;
  }

  return status;
}

} // namespace ranally

