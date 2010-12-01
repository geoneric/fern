#ifndef INCLUDED_RANALLY_THREADVISITORTEST
#define INCLUDED_RANALLY_THREADVISITORTEST

#include "AlgebraParser.h"
#include "XmlParser.h"
#include "ThreadVisitor.h"



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class ThreadVisitorTest
{

private:

  ranally::AlgebraParser _algebraParser;
  ranally::XmlParser _xmlParser;
  ranally::ThreadVisitor _visitor;

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

};

#endif
