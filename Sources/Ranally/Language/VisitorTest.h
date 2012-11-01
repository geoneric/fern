#pragma once
#include "Ranally/Language/AlgebraParser.h"
#include "Ranally/Language/XmlParser.h"


namespace boost {
    namespace unit_test {
        class test_suite;
    }
}


class VisitorTest
{

public:

                   VisitorTest         ();

    void           testCountVerticesVisitor();

    static boost::unit_test::test_suite* suite();

private:

  ranally::AlgebraParser _algebraParser;

  ranally::XmlParser _xmlParser;

};
