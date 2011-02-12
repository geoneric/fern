#ifndef INCLUDED_RANALLY_LANGUAGE_ALGEBRAPARSERTEST
#define INCLUDED_RANALLY_LANGUAGE_ALGEBRAPARSERTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class AlgebraParserTest
{

private:

public:

                   AlgebraParserTest   ();

  void             testParseEmptyScript();

  void             testParseName       ();

  void             testParseAssignment ();

  void             testParseFile       ();

  void             testParseString     ();

  void             testParseNumber     ();

  void             testParseCall       ();

  void             testParseUnaryOperator();

  void             testParseBinaryOperator();

  void             testParseBooleanOperator();

  void             testParseComparisonOperator();

  void             testParseMultipleStatements();

  void             testParseIf         ();

  void             testParseWhile      ();

  static boost::unit_test::test_suite* suite();

};

#endif
