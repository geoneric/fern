#pragma once
#include "Ranally/Language/AlgebraParser.h"
#include "Ranally/Language/XmlParser.h"
#include "Ranally/Language/AnnotateVisitor.h"



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class AnnotateVisitorTest
{

public:

                   AnnotateVisitorTest ();

  void             testVisitEmptyScript();

  void             testVisitNumber     ();

  void             testVisitOperation  ();

  static boost::unit_test::test_suite* suite();

private:

  ranally::language::AlgebraParser _algebraParser;
  ranally::language::XmlParser _xmlParser;
  ranally::language::AnnotateVisitor _visitor;

};
