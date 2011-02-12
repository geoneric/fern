#ifndef INCLUDED_RANALLY_NAMEVERTEXTEST
#define INCLUDED_RANALLY_NAMEVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class NameVertexTest
{

private:

public:

                   NameVertexTest      ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
