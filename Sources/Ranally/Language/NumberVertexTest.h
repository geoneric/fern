#ifndef INCLUDED_RANALLY_NUMBERVERTEXTEST
#define INCLUDED_RANALLY_NUMBERVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class NumberVertexTest
{

private:

public:

                   NumberVertexTest    ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
