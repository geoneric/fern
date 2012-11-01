#pragma once
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

    void           testVisitEmptyScript();

    void           testVisitName       ();

    void           testVisitAssignment ();

    void           testVisitString     ();

    void           testVisitNumber     ();

    void           testVisitFunction   ();

    void           testVisitOperator   ();

    void           testVisitMultipleStatements();

    void           testVisitIf         ();

    void           testVisitWhile      ();

    static boost::unit_test::test_suite* suite();

private:

    ranally::AlgebraParser _algebraParser;

    ranally::XmlParser _xmlParser;

    // ranally::DotVisitor _visitor;

};
