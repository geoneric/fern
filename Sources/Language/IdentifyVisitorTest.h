#ifndef INCLUDED_RANALLY_IDENTIFYVISITORTEST
#define INCLUDED_RANALLY_IDENTIFYVISITORTEST

#include "AlgebraParser.h"
#include "XmlParser.h"
#include "IdentifyVisitor.h"



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class IdentifyVisitorTest
{

private:

  ranally::AlgebraParser _algebraParser;
  ranally::XmlParser _xmlParser;
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
