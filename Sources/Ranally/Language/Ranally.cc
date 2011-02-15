#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include <boost/algorithm/string/join.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_lit.hpp>
#include "dev_UnicodeUtils.h"
#include "Ranally/Configure.h"
#include "Ranally/Language/AlgebraParser.h"
#include "Ranally/Language/AstDotVisitor.h"
#include "Ranally/Language/IdentifyVisitor.h"
#include "Ranally/Language/XmlParser.h"
#include "Ranally/Language/ScriptVertex.h"
#include "Ranally/Language/ScriptVisitor.h"
#include "Ranally/Language/ThreadVisitor.h"









// TODO Think of all tasks this utility needs to perform and design a
//      command line interface for that.
//      - interpret/execute script
//      - convert script to script (roundtrip)
//      - convert script to dot-graph
//      - convert script to C++
//      - convert script to C++, including Python extension
// Some of these actions can happen simultaneously. Others are silly to
// combine.
//   valid: all conversions
//   silly: interpret and convert

// Ideas:
// - Execute is default/implied if no conversion action is provided.
// - Convert argument has a value of script | dot | C++ | Python
// - Iff convert argument is provided, than no execution takes place.
//   Positional arguments are related to the conversion process
//   (target dir / file, ...).
//
// - execute command is implied.
// - first positional is filename of script.
// - if --script option is provided, than script positional should not be.
//   ranally model.ran
//   ranally execute model.ran
//   ranally --script "slope = slope("dem")"
//   ranally execute --script "slope = slope("dem")"
//
// - first positional is language argument.
// - 
//   ranally convert dot model.ran model.dot
//   ranally convert c++ model.ran
//   ranally convert c++ --namespace bla --output-directory ./blo model.ran
//   ranally convert python --package bli model.ran
//
//   The Python extension converter should not create the core C++ code.
//   The c++ converter is for doing that. Or can we use two convert commands?



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



void showConvertHelp()
{
  std::cout <<
    "usage: ranally convert LANGUAGE SCRIPT\n"
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
    "usage: ranally convert dot [input script] [output script]\n"
    "\n"
    "Convert the script to a dot graph containing the syntax tree.\n"
    "\n"
    "  input script        Script to convert (standard input is read if no\n"
    "                      script arguments are provided)\n"
    "  output script       File to write result to (standard output is\n"
    "                      written if this argument is not provided)\n"
    ;
}



void showConvertRanallyHelp()
{
  std::cout <<
    "usage: ranally convert ranally [input script] [output script]\n"
    "\n"
    "Convert the script to a ranally script (round-trip).\n"
    "\n"
    "  input script        Script to convert (standard input is read if no\n"
    "                      script arguments are provided)\n"
    "  output script       File to write result to (standard output is\n"
    "                      written if this argument is not provided)\n"
    ;
}



void showExecuteHelp()
{
  std::cout <<
    "usage: ranally [execute] SCRIPT\n"
    "\n"
    "Execute the script. The execute command is implied if not provided.\n"
    ;
}



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
private:

  std::string      _commandLine;

protected:

  Command(
    std::string commandLine)
    : _commandLine(commandLine)
  {
  }

  std::string const& commandLine() const
  {
    return _commandLine;
  }

public:

  virtual int      execute             ()=0;

};



class ConvertCommand: public Command
{
private:

protected:

public:

  ConvertCommand(
    std::string const& commandLine)
    : Command(commandLine)
  {
  }

  int convertToCpp(
    std::string const& /* remainder */)
  {
    int status = EXIT_SUCCESS;
    std::cout << "Conversion to C++ not supported yet\n";
    return status;
  }

  int convertToDot(
    std::string const& remainder)
  {
    namespace bs = boost::spirit;
    namespace bp = boost::phoenix;
    using bs::qi::_1;

    // TODO type = ast | flowgraph
    std::string option;
    // std::vector<char> outputFileName, inputFileName;
    std::string commandLine(remainder);
    std::string::iterator first(commandLine.begin());
    std::string::iterator last(commandLine.end());

    // text %= bs::qi::alnum;
    typedef std::string::const_iterator Iterator;
    typedef bs::ascii::space_type SpaceType;
    bs::qi::rule<Iterator, std::string(), SpaceType> outputFileName;
    // outputFileName = bs::lexeme[+bs::qi::alnum[_val += _1]];
    // outputFileName = (+bs::qi::alnum)[_val += _1];

    // bool result = bs::qi::phrase_parse(first, last,
    //   bs::lit(std::string("--help"))[bp::ref(option) = "help"] |
    //   (
    //     -(
    //       bs::lit(std::string("--output")) >>
    //       (+bs::qi::alnum)[_val += _1]
    //     ) >>
    //     -(
    //       (+bs::qi::alnum)[bp::ref(inputFileName) = _1]
    //     )
    //   )
    //   , boost::spirit::ascii::space);

    int status = EXIT_SUCCESS;

    // if(!result) {
    //   std::cerr << "Error while parsing command line\n";
    //   std::cerr << "See 'ranally convert dot --help' for usage information.\n";
    //   status = EXIT_FAILURE;
    // }
    // else {
    //   std::cout << inputFileName << std::endl;
    //   std::cout << outputFileName << std::endl;
    //   if(option == "help") {
    //     showConvertDotHelp();
    //   }
    //   else {
    //     assert(option.empty());

    //     ranally::AstDotVisitor astDotVisitor;
    //     ranally::language::ThreadVisitor threadVisitor;
    //     ranally::language::IdentifyVisitor identifyVisitor;
    //     ranally::language::AlgebraParser parser;
    //     UnicodeString xml;

    //     if(inputFileName.empty()) {
    //       // Read script from the standard input stream.
    //       std::ostringstream script;
    //       script << std::cin.rdbuf();
    //       xml = ranally::language::AlgebraParser().parseString(UnicodeString(
    //         script.str().c_str()));
    //     }
    //     else {
    //       // Read script from a file.
    //       xml = ranally::language::AlgebraParser().parseFile(UnicodeString(
    //         inputFileName.c_str()));
    //     }

    //     boost::shared_ptr<ranally::language::ScriptVertex> tree(
    //       ranally::language::XmlParser().parse(xml));
    //     tree->Accept(threadVisitor);
    //     tree->Accept(identifyVisitor);
    //     tree->Accept(astDotVisitor);

    //     if(outputFileName.empty()) {
    //       std::cout << dev::encodeInUTF8(astDotVisitor.script()) << std::endl;
    //     }
    //     else {
    //       std::cout << "TODO write to file\n";
    //     }
    //   }
    // }

    return status;
  }

  int convertToPython(
    std::string const& /* remainder */)
  {
    int status = EXIT_SUCCESS;
    std::cout << "Conversion to Python not supported yet\n";
    return status;
  }

  int convertToRanally(
    std::string const& /* remainder */)
  {
    int status = EXIT_SUCCESS;
    return status;
  }

  int execute()
  {
    namespace bs = boost::spirit;
    namespace bp = boost::phoenix;

    std::string option, language;
    std::string commandLine(this->commandLine());
    std::string::iterator first(commandLine.begin());
    std::string::iterator last(commandLine.end());
    bool result = bs::qi::phrase_parse(first, last,
      bs::lit(std::string("--help"))[bp::ref(option) = "help"] |
      (
        bs::lit(std::string("cpp"))[bp::ref(language) = "cpp"] |
        bs::lit(std::string("dot"))[bp::ref(language) = "dot"] |
        bs::lit(std::string("python"))[bp::ref(language) = "python"] |
        bs::lit(std::string("ranally"))[bp::ref(language) = "ranally"]
      )
      , boost::spirit::ascii::space);

    int status = EXIT_SUCCESS;

    if(!result) {
      std::cerr << "Error while parsing command line\n";
      std::cerr << "See 'ranally convert --help' for usage information.\n";
      status = EXIT_FAILURE;
    }
    else {
      if(option == "help") {
        showConvertHelp();
      }
      else {
        assert(option.empty());
        assert(!language.empty());

        std::string remainder(first, last);

        if(language == "cpp") {
          status = convertToCpp(remainder);
        }
        else if(language == "dot") {
          status = convertToDot(remainder);
        }
        else if(language == "python") {
          status = convertToPython(remainder);
        }
        else if(language == "ranally") {
          status = convertToRanally(remainder);
        }
        else {
          assert(false);
        }
      }
    }

    return status;
  }

};

} // Anonymous namespace



int main(
  int argc,
  char** argv)
{
  if(true) {
    // Convert arguments to string.
    std::vector<std::string> strings(argv, argv + argc);
    std::string string = boost::algorithm::join(strings, " ");

    namespace bs = boost::spirit;
    namespace bp = boost::phoenix;

    std::string option;
    std::string commandName;
    std::string::iterator first(string.begin());
    std::string::iterator last(string.end());
    bool result = bs::qi::phrase_parse(first, last,
      boost::spirit::lit(std::string(argv[0])) >>
      (
        bs::lit(std::string("--help"))[bp::ref(option) = "help"] |
        bs::lit(std::string("--build"))[bp::ref(option) = "build"] |
        bs::lit(std::string("--version"))[bp::ref(option) = "version"] |
        (
          bs::lit(std::string("execute"))[bp::ref(commandName) = "execute"] |
          bs::lit(std::string("convert"))[bp::ref(commandName) = "convert"]
        )
      )
      , boost::spirit::ascii::space);

    int status = EXIT_SUCCESS;

    if(!result) {
      std::cerr << "Error while parsing command line\n";
      std::cerr << "See 'ranally --help' for usage information.\n";
      status =  EXIT_FAILURE;
    }
    else {
      if(option == "help") {
        showGeneralHelp();
      }
      else if(option == "build") {
        showBuild();
      }
      else if(option == "version") {
        showVersion();
      }
      else {
        assert(option.empty());
        assert(!commandName.empty());

        std::string remainder(first, last);
        boost::scoped_ptr<Command> command;

        if(commandName == "convert") {
          command.reset(new ConvertCommand(remainder));
        }
        else if(commandName == "execute") {
          // TODO
        }
        else {
          assert(false);
        }

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
  }

  //   std::string command;

  //   namespace bs = boost::spirit;
  //   namespace bp = boost::phoenix;
  //   using bs::qi::_1;

  //   std::string::iterator first(string.begin());
  //   std::string::iterator last(string.end());
  //   bool result = bs::qi::phrase_parse(first, last,
  //     boost::spirit::lit(std::string(argv[0])) >>
  //     (
  //       bs::lit(std::string("--help")) |
  //       bs::lit(std::string("--build")) |
  //       bs::lit(std::string("--version")) |
  //       (
  //         bs::lit(std::string("execute")) >>
  //         +bs::qi::alnum
  //       ) |
  //       (
  //         bs::lit(std::string("convert"))[bp::ref(command) = "convert"] >>
  //         (
  //           (
  //             bs::lit(std::string("ranally")) >>
  //             bs::lit(std::string("--output")) >>
  //             +bs::qi::alnum
  //           ) |
  //           (
  //             bs::lit(std::string("dot")) >>
  //             -bs::lit(std::string("--output")) >>
  //             -(+bs::qi::alnum)
  //           ) |
  //           bs::lit(std::string("cpp")) |
  //           bs::lit(std::string("python"))
  //         )
  //       )
  //     )
  //     , boost::spirit::ascii::space);

  //   if(!result || first != last) {
  //     std::cout << "failure parsing!" << std::endl;
  //     return EXIT_FAILURE;
  //   }
  //   else {
  //     std::cout << "success parsing!" << std::endl;
  //     std::cout << "command: " << command << std::endl;
  //     return EXIT_SUCCESS;
  //   }
  // }





  if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
    // No arguments, or the help option.
    showGeneralHelp();
    return EXIT_SUCCESS;
  }
  else if(std::strcmp(argv[1], "--version") == 0) {
    showVersion();
    return EXIT_SUCCESS;
  }
  else if(std::strcmp(argv[1], "--build") == 0) {
    showBuild();
    return EXIT_SUCCESS;
  }
  else if(std::strcmp(argv[1], "convert") == 0) {
    if(argc == 2 || std::strcmp(argv[2], "--help") == 0) {
      // No arguments, or the help option.
      showConvertHelp();
      return EXIT_SUCCESS;
    }

    assert(argc >= 3);

    if(std::strcmp(argv[2], "dot") == 0) {
      if(argc >= 4 && std::strcmp(argv[3], "--help") == 0) {
        showConvertDotHelp();
        return EXIT_SUCCESS;
      }

      ranally::AstDotVisitor astDotVisitor;
      ranally::language::ThreadVisitor threadVisitor;
      ranally::language::IdentifyVisitor identifyVisitor;
      ranally::language::AlgebraParser parser;
      UnicodeString xml;

      if(argc == 3) {
        // Read script from the standard input stream.
        std::ostringstream script;
        script << std::cin.rdbuf();
        xml = ranally::language::AlgebraParser().parseString(UnicodeString(
          script.str().c_str()));
      }
      else if(argc == 4) {
        // Read script from a file.
        std::string inputFileName(argv[3]);
        xml = ranally::language::AlgebraParser().parseFile(UnicodeString(
          inputFileName.c_str()));
      }

      boost::shared_ptr<ranally::language::ScriptVertex> tree(
        ranally::language::XmlParser().parse(xml));
      tree->Accept(threadVisitor);
      tree->Accept(identifyVisitor);
      tree->Accept(astDotVisitor);

      if(argc <= 4) {
        std::cout << dev::encodeInUTF8(astDotVisitor.script()) << std::endl;
      }
      else if(argc == 5) {
        std::cout << dev::encodeInUTF8(astDotVisitor.script()) << std::endl;
        std::cout << "TODO write to file\n";
      }

      return EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[2], "c++") == 0) {
      std::cout << "Convert to c++...\n";
      return EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[2], "python") == 0) {
      std::cout << "Convert to python...\n";
      return EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[2], "ranally") == 0) {
      if(argc >= 4 && std::strcmp(argv[3], "--help") == 0) {
        showConvertRanallyHelp();
        return EXIT_SUCCESS;
      }

      ranally::ScriptVisitor visitor;
      ranally::language::AlgebraParser parser;
      UnicodeString xml;

      if(argc == 3) {
        // Read script from the standard input stream.
        std::ostringstream script;
        script << std::cin.rdbuf();
        xml = ranally::language::AlgebraParser().parseString(UnicodeString(
          script.str().c_str()));
      }
      else if(argc == 4) {
        // Read script from a file.
        std::string inputFileName(argv[3]);
        xml = ranally::language::AlgebraParser().parseFile(UnicodeString(
          inputFileName.c_str()));
      }

      ranally::language::XmlParser().parse(xml)->Accept(visitor);

      if(argc <= 4) {
        std::cout << dev::encodeInUTF8(visitor.script()) << std::endl;
      }
      else if(argc == 5) {
        std::cout << dev::encodeInUTF8(visitor.script()) << std::endl;
        std::cout << "TODO write to file\n";
      }

      return EXIT_SUCCESS;
    }
    else {
      std::cerr << "Unknown target language...\n";
      std::cerr << "Conversion help...\n";
      return EXIT_FAILURE;
    }
  }
  else if(std::strcmp(argv[1], "execute") == 0) {
    if(argc == 2 || std::strcmp(argv[2], "--help") == 0) {
      // No arguments, or the help option.
      showExecuteHelp();
      return EXIT_SUCCESS;
    }

    std::cout << "Execute script...\n";
    return EXIT_SUCCESS;
  }
  else {
    std::cout << "Execute script...\n";
    return EXIT_SUCCESS;
  }





  // po::options_description optionsDescription("Allowed options");
  // optionsDescription.add_options()
  //   ("help", "show help message")
  //   ("build", "show build info")
  //   ("version", "show version")
  //   ;

  // po::variables_map variablesMap;
  // po::store(po::parse_command_line(argc, argv, optionsDescription),
  //   variablesMap);
  // po::notify(variablesMap);

  // if(variablesMap.empty() || variablesMap.count("help")) {
  //   std::cout << optionsDescription << "\n";
  //   return EXIT_SUCCESS;
  // }

  // if(variablesMap.count("version")) {
  //   std::cout << "ranally " << RANALLY_VERSION << "-" << RANALLY_BUILD_STAGE
  //     << "\n";
  //   std::cout << RANALLY_COPYRIGHT << "\n";
  //   return EXIT_SUCCESS;
  // }

  // if(variablesMap.count("build")) {
  //   std::cout << RANALLY_BUILD_TYPE << " build (" << __DATE__ << ")\n"
  //     << RANALLY_ARCHITECTURE << ", "
  //     << RANALLY_SYSTEM << ", "
  //     << RANALLY_CXX_COMPILER
  //     << "\n";
  //   return EXIT_SUCCESS;
  // }

  return EXIT_SUCCESS;
}