#ifndef INCLUDED_RANALLY_LANGUAGE_THREADVISITORTEST
#define INCLUDED_RANALLY_LANGUAGE_THREADVISITORTEST

#include "Ranally/Language/AlgebraParser.h"
#include "Ranally/Language/XmlParser.h"
#include "Ranally/Language/ThreadVisitor.h"



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class ThreadVisitorTest
{

public:

                   ThreadVisitorTest   ();

  void             testVisitEmptyScript();

  void             testVisitName       ();

  void             testVisitAssignment ();

  void             testVisitString     ();

  void             testVisitNumber     ();

  void             testVisitFunction   ();

  void             testVisitOperator   ();

  void             testVisitNestedExpression();

  void             testVisitMultipleStatements();

  void             testVisitIf         ();

  void             testVisitWhile      ();

  static boost::unit_test::test_suite* suite();

private:

  ranally::language::AlgebraParser _algebraParser;
  ranally::language::XmlParser _xmlParser;
  ranally::language::ThreadVisitor _visitor;

};

#endif
