#pragma once
#include "Ranally/Interpreter/Interpreter.h"
#include "Ranally/Language/OptimizeVisitor.h"
#include "Ranally/Language/ScriptVisitor.h"


namespace boost {
    namespace unit_test {
        class test_suite;
    }
}


class OptimizeVisitorTest
{

public:

                   OptimizeVisitorTest ();

    void           testRemoveTemporaryIdentifier();

    static boost::unit_test::test_suite* suite();

private:

    ranally::Interpreter _interpreter;

    ranally::ScriptVisitor _scriptVisitor;

    ranally::OptimizeVisitor _optimizeVisitor;

};
