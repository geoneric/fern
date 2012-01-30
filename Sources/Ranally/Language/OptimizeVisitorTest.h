#ifndef INCLUDED_RANALLY_LANGUAGE_OPTIMIZEVISITORTEST
#define INCLUDED_RANALLY_LANGUAGE_OPTIMIZEVISITORTEST

#include "Ranally/Interpreter/Interpreter.h"
#include "Ranally/Language/OptimizeVisitor.h"
#include "Ranally/Language/ScriptVisitor.h"



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class OptimizeVisitorTest
{

public:

                   OptimizeVisitorTest ();

  void             testRemoveTemporaryIdentifier();

  static boost::unit_test::test_suite* suite();

  ranally::interpreter::Interpreter _interpreter;
  ranally::language::ScriptVisitor _scriptVisitor;
  ranally::language::OptimizeVisitor _optimizeVisitor;

};

#endif
