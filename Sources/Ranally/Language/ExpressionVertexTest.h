#ifndef INCLUDED_RANALLY_LANGUAGE_EXPRESSIONVERTEXTEST
#define INCLUDED_RANALLY_LANGUAGE_EXPRESSIONVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class ExpressionVertexTest
{

public:

                   ExpressionVertexTest();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
