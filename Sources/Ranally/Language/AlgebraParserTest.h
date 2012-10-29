#pragma once


namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class AlgebraParserTest
{

public:

                   AlgebraParserTest   ();

  void             testParseEmptyScript();

  void             testParseName       ();

  void             testParseAssignment ();

  void             testParseFile       ();

  void             testParseString     ();

  void             testParseNumber     ();

  void             testParseCall       ();

  // void             testParsePrint      ();

  void             testParseUnaryOperator();

  void             testParseBinaryOperator();

  void             testParseBooleanOperator();

  void             testParseComparisonOperator();

  void             testParseMultipleStatements();

  void             testParseIf         ();

  void             testParseWhile      ();

  static boost::unit_test::test_suite* suite();

};
