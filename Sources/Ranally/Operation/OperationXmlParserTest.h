#pragma once


namespace boost {
    namespace unit_test {
        class test_suite;
    }
}


class OperationXmlParserTest
{

public:

                   OperationXmlParserTest       ();

    void           testParse           ();

    static boost::unit_test::test_suite* suite();

private:

};
