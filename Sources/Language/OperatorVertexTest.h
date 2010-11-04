#ifndef INCLUDED_RANALLY_OPERATORVERTEXTEST
#define INCLUDED_RANALLY_OPERATORVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class OperatorVertexTest
{

private:

public:

                   OperatorVertexTest  ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
