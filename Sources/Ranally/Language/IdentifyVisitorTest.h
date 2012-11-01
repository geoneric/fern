#pragma once
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

public:

                   IdentifyVisitorTest ();

    void           testVisitEmptyScript();

    void           testVisitName       ();

    void           testVisitAssignment ();

    void           testVisitIf         ();

    void           testVisitReuseOfIdentifiers();

    static boost::unit_test::test_suite* suite();

private:

  ranally::AlgebraParser _algebraParser;

  ranally::XmlParser _xmlParser;

  ranally::IdentifyVisitor _visitor;

};
