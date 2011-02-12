#ifndef INCLUDED_RANALLY_LANGUAGE_IDENTIFYVISITORTEST
#define INCLUDED_RANALLY_LANGUAGE_IDENTIFYVISITORTEST

#include "Ranally/Language/AlgebraParser.h"
#include "Ranally/Language/XmlParser.h"
#include "Ranally/Language/IdentifyVisitor.h"



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class IdentifyVisitorTest
{

private:

  ranally::language::AlgebraParser _algebraParser;
  ranally::language::XmlParser _xmlParser;
  ranally::language::IdentifyVisitor _visitor;

public:

                   IdentifyVisitorTest ();

  void             testVisitEmptyScript();

  void             testVisitName       ();

  void             testVisitAssignment ();

  void             testVisitIf         ();

  static boost::unit_test::test_suite* suite();

};

#endif
