#include "ExecuteCommand.h"
#include "Ranally/Operation/XmlParser.h"
#include "Ranally/Operation/Operation-xml.h"
#include "Ranally/Language/AnnotateVisitor.h"
#include "Ranally/Language/ExecuteVisitor.h"
#include "Ranally/Language/IdentifyVisitor.h"
#include "Ranally/Language/ScriptVertex.h"
#include "Ranally/Language/ThreadVisitor.h"
#include "Ranally/Language/ValidateVisitor.h"
#include "Ranally/Language/XmlParser.h"



namespace ranally {
namespace {

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

} // Anonymous namespace



ExecuteCommand::ExecuteCommand(
  int argc,
  char** argv)

  : Command(argc, argv)

{
}



ExecuteCommand::~ExecuteCommand()
{
}



void ExecuteCommand::execute(
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

  ranally::language::ValidateVisitor validateVisitor;
  tree->Accept(validateVisitor);

  ranally::language::ExecuteVisitor executeVisitor;
  tree->Accept(executeVisitor);
}



int ExecuteCommand::execute()
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
      ? argv()[1] : "";
    UnicodeString xml = read(inputFileName);
    execute(xml);
    status = EXIT_SUCCESS;
  }

  return status;
}

} // namespace ranally

