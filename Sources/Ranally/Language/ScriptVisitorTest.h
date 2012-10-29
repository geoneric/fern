#pragma once
#include "Ranally/Language/AlgebraParser.h"
#include "Ranally/Language/XmlParser.h"
#include "Ranally/Language/ScriptVisitor.h"


namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class ScriptVisitorTest
{

public:

                   ScriptVisitorTest   ();

  void             testVisitEmptyScript();

  void             testVisitName       ();

  void             testVisitAssignment ();

  void             testVisitString     ();

  void             testVisitNumber     ();

  void             testVisitFunction   ();

  void             testVisitOperator   ();

  void             testVisitMultipleStatements();

  void             testVisitIf         ();

  void             testVisitWhile      ();

  static boost::unit_test::test_suite* suite();

private:

  ranally::language::AlgebraParser _algebraParser;
  ranally::language::XmlParser _xmlParser;
  ranally::language::ScriptVisitor _visitor;

};
