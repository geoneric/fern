#ifndef INCLUDED_RANALLY_DOTVISITORTEST
#define INCLUDED_RANALLY_DOTVISITORTEST

#include "AlgebraParser.h"
#include "XmlParser.h"
#include "DotVisitor.h"



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class DotVisitorTest
{

private:

  ranally::AlgebraParser _algebraParser;
  ranally::XmlParser _xmlParser;
  // ranally::DotVisitor _visitor;

public:

                   DotVisitorTest      ();

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
