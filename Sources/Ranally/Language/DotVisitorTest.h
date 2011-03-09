#ifndef INCLUDED_RANALLY_LANGUAGE_DOTVISITORTEST
#define INCLUDED_RANALLY_LANGUAGE_DOTVISITORTEST

#include "Ranally/Language/AlgebraParser.h"
#include "Ranally/Language/XmlParser.h"
#include "Ranally/Language/DotVisitor.h"



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class DotVisitorTest
{

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

private:

  ranally::language::AlgebraParser _algebraParser;
  ranally::language::XmlParser _xmlParser;
  // ranally::language::DotVisitor _visitor;

};

#endif
