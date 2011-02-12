#ifndef INCLUDED_RANALLY_WHILEVERTEXTEST
#define INCLUDED_RANALLY_WHILEVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class WhileVertexTest
{

private:

public:

                   WhileVertexTest     ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
