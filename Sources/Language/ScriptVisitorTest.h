#ifndef INCLUDED_RANALLY_SCRIPTVISITORTEST
#define INCLUDED_RANALLY_SCRIPTVISITORTEST

#include "AlgebraParser.h"
#include "XmlParser.h"
#include "ScriptVisitor.h"



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class ScriptVisitorTest
{

private:

  ranally::AlgebraParser _algebraParser;
  ranally::XmlParser _xmlParser;
  ranally::ScriptVisitor _visitor;

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

};

#endif
