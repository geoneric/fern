#pragma once
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

    void           testVisitEmptyScript();

    void           testVisitName       ();

    void           testVisitAssignment ();

    void           testVisitString     ();

    void           testVisitNumber     ();

    void           testVisitFunction   ();

    void           testVisitOperator   ();

    void           testVisitNestedExpression();

    void           testVisitMultipleStatements();

    void           testVisitIf         ();

    void           testVisitWhile      ();

    static boost::unit_test::test_suite* suite();

private:

    ranally::AlgebraParser _algebraParser;

    ranally::XmlParser _xmlParser;

    ranally::ThreadVisitor _visitor;

};
