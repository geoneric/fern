#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/scoped_ptr.hpp>
#include "dev_UnicodeUtils.h"
#include "Ranally/Configure.h"
#include "Ranally/Language/AlgebraParser.h"
#include "Ranally/Language/AnnotateVisitor.h"
#include "Ranally/Language/AstDotVisitor.h"
#include "Ranally/Language/ExecuteVisitor.h"
#include "Ranally/Language/FlowgraphDotVisitor.h"
#include "Ranally/Language/IdentifyVisitor.h"
#include "Ranally/Language/XmlParser.h"
#include "Ranally/Language/ScriptVertex.h"
#include "Ranally/Language/ScriptVisitor.h"
#include "Ranally/Language/ThreadVisitor.h"
#include "Ranally/Language/ValidateVisitor.h"
#include "Ranally/Operation/XmlParser.h"
#include "Ranally/Operation/Operation-xml.h"



namespace {

void showGeneralHelp()
{
  std::cout <<
    "usage: ranally [--help] [--build] [--version] [COMMAND] [ARGS]\n"
    "\n"
    "--help                Show help message\n"
    "--version             Show version\n"
    "--build               Show build info\n"
    "\n"
    "commands:\n"
    "  execute             Execute script (this is the default when no\n"
    "                      command is provided)\n"
    "  convert             Convert script\n"
    "\n"
    "See 'ranally COMMAND --help' for more information on a specific command.\n"
    ;
}



void showExecuteHelp()
{
  std::cout <<
    "usage: ranally execute [--help] INPUT_SCRIPT\n"
    "\n"
    "Execute the script.\n"
    "\n"
    "  INPUT_SCRIPT        Script to execute or - to read from standard input\n"
    ;
}



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



void showVersion()
{
  std::cout << "ranally " << RANALLY_VERSION << "-" << RANALLY_BUILD_STAGE
    << "\n";
  std::cout << RANALLY_COPYRIGHT << "\n";
}



void showBuild()
{
  std::cout << RANALLY_BUILD_TYPE << " build (" << __DATE__ << ")\n"
    << RANALLY_ARCHITECTURE << ", "
    << RANALLY_SYSTEM << ", "
    << RANALLY_CXX_COMPILER
    << "\n";
}



class Command
{

public:

  virtual int      execute             ()=0;

protected:

  Command(
    std::string commandLine)
    : _commandLine(commandLine)
  {
  }

  Command(
    int argc,
    char** argv)
    : _argc(argc),
      _argv(argv)
  {
  }

  std::string const& commandLine() const
  {
    return _commandLine;
  }

  int argc() const
  {
    return _argc;
  }

  char** argv() const
  {
    return _argv;
  }

private:

  std::string      _commandLine;

  int              _argc;

  char**           _argv;

};



class ConvertCommand:
  public Command
{

public:

  ConvertCommand(
    std::string const& commandLine)
    : Command(commandLine)
  {
  }

  ConvertCommand(
    int argc,
    char** argv)
    : Command(argc, argv)
  {
  }

  int convertToRanally(
    int /* argc */,
    char** /* argv */)
  {
    std::cout << "Conversion to Ranally script not supported yet\n";
    return EXIT_SUCCESS;
  }

  int convertToCpp(
    int /* argc */,
    char** /* argv */)
  {
    std::cout << "Conversion to C++ not supported yet\n";
    return EXIT_SUCCESS;
  }

  int convertToDotAst(
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

        ranally::AstDotVisitor astDotVisitor(modes);
        ranally::language::ThreadVisitor threadVisitor;
        ranally::language::IdentifyVisitor identifyVisitor;
        ranally::language::AlgebraParser parser;
        UnicodeString xml;

        if(inputFileName.empty()) {
          // Read script from the standard input stream.
          std::ostringstream script;
          script << std::cin.rdbuf();
          xml = ranally::language::AlgebraParser().parseString(UnicodeString(
            script.str().c_str()));
        }
        else {
          // Read script from a file.
          xml = ranally::language::AlgebraParser().parseFile(UnicodeString(
            inputFileName.c_str()));
        }

        boost::shared_ptr<ranally::language::ScriptVertex> tree(
          ranally::language::XmlParser().parse(xml));
        tree->Accept(threadVisitor);
        tree->Accept(identifyVisitor);
        tree->Accept(astDotVisitor);

        std::string result = dev::encodeInUTF8(astDotVisitor.script());

        if(outputFileName.empty()) {
          std::cout << result;
        }
        else {
          std::ofstream file(outputFileName.c_str());
          file << result;
        }

        status = EXIT_SUCCESS;
      }
    }

    return status;
  }

  int convertToDotFlowgraph(
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

        ranally::FlowgraphDotVisitor flowgraphDotVisitor;
        ranally::language::ThreadVisitor threadVisitor;
        ranally::language::IdentifyVisitor identifyVisitor;
        ranally::language::AlgebraParser parser;
        UnicodeString xml;

        if(inputFileName.empty()) {
          // Read script from the standard input stream.
          std::ostringstream script;
          script << std::cin.rdbuf();
          xml = ranally::language::AlgebraParser().parseString(UnicodeString(
            script.str().c_str()));
        }
        else {
          // Read script from a file.
          xml = ranally::language::AlgebraParser().parseFile(UnicodeString(
            inputFileName.c_str()));
        }

        boost::shared_ptr<ranally::language::ScriptVertex> tree(
          ranally::language::XmlParser().parse(xml));
        tree->Accept(threadVisitor);
        tree->Accept(identifyVisitor);
        tree->Accept(flowgraphDotVisitor);

        std::string result = dev::encodeInUTF8(flowgraphDotVisitor.script());

        if(outputFileName.empty()) {
          std::cout << result;
        }
        else {
          std::ofstream file(outputFileName.c_str());
          file << result;
        }
      }

      status = EXIT_SUCCESS;
    }

    return status;
  }

  int convertToDot(
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

  int convertToPython(
    int /* argc */,
    char** /* argv */)
  {
    std::cout << "Conversion to Python not supported yet\n";
    return EXIT_SUCCESS;
  }

  int execute()
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
    else {
      std::cerr << "Unknown target language: " << argv()[1] << "\n";
      std::cerr << "See 'ranally convert --help' for list of languages.\n";
      status = EXIT_FAILURE;
    }

    return status;
  }

};



class ExecuteCommand:
  public Command
{

public:

  ExecuteCommand(
    int argc,
    char** argv)

    : Command(argc, argv)

  {
  }

  void execute(
    UnicodeString const& xml)
  {
    boost::shared_ptr<ranally::language::ScriptVertex> tree;
    ranally::operation::OperationsPtr operations;

    {
      ranally::language::XmlParser xmlParser;
      tree = xmlParser.parse(xml);
    }

    {
      ranally::operation::XmlParser xmlParser;
      operations = xmlParser.parse(xml);
    }

    ranally::language::ThreadVisitor threadVisitor;
    tree->Accept(threadVisitor);

    ranally::language::IdentifyVisitor identifyVisitor;
    tree->Accept(identifyVisitor);

    ranally::language::AnnotateVisitor annotateVisitor;
    tree->Accept(annotateVisitor);

    ranally::language::ValidateVisitor validateVisitor(operations);
    tree->Accept(validateVisitor);

    ranally::language::ExecuteVisitor executeVisitor;
    tree->Accept(executeVisitor);
  }

  int execute()
  {
    int status = EXIT_FAILURE;

    if(argc() == 1 || std::strcmp(argv()[1], "--help") == 0) {
      // No arguments, or the help option.
      showExecuteHelp();
      status = EXIT_SUCCESS;
    }
    else if(argc() > 2) {
      std::cerr << "Too many arguments.\n";
      showExecuteHelp();
      status = EXIT_FAILURE;
    }
    else {
      std::string inputFileName = std::strcmp(argv()[1], "-") != 0
        ? argv()[1]
        : "";

      UnicodeString xml;

      if(inputFileName.empty()) {
        // Read script from the standard input stream.
        std::ostringstream script;
        script << std::cin.rdbuf();
        xml = ranally::language::AlgebraParser().parseString(UnicodeString(
          script.str().c_str()));
      }
      else {
        // Read script from a file.
        xml = ranally::language::AlgebraParser().parseFile(UnicodeString(
          inputFileName.c_str()));
      }

      execute(xml);
      status = EXIT_SUCCESS;
    }

    return status;
  }

};

} // Anonymous namespace



int main(
  int argc,
  char** argv)
{
  int status = EXIT_FAILURE;

  if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
    // No arguments, or the help option.
    showGeneralHelp();
    status = EXIT_SUCCESS;
  }
  else if(std::strcmp(argv[1], "--version") == 0) {
    showVersion();
    status = EXIT_SUCCESS;
  }
  else if(std::strcmp(argv[1], "--build") == 0) {
    showBuild();
    status = EXIT_SUCCESS;
  }
  else {
    boost::scoped_ptr<Command> command;

    // A command may be given. Find out which one.
    if(std::strcmp(argv[1], "convert") == 0) {
      command.reset(new ConvertCommand(argc - 1, argv + 1));
    }
    else if(std::strcmp(argv[1], "execute") == 0) {
      command.reset(new ExecuteCommand(argc - 1, argv + 1));
    }
    else {
      // Default command is 'execute'.
      command.reset(new ExecuteCommand(argc, argv));
    }

    assert(command);

    try {
      status = command->execute();
    }
    catch(std::exception const& exception) {
      std::cerr << exception.what() << '\n';
      status = EXIT_FAILURE;
    }
  }

  return status;
}

