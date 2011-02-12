#ifndef INCLUDED_RANALLY_STRINGVERTEXTEST
#define INCLUDED_RANALLY_STRINGVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class StringVertexTest
{

private:

public:

                   StringVertexTest    ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
