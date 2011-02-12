#ifndef INCLUDED_RANALLY_IFVERTEXTEST
#define INCLUDED_RANALLY_IFVERTEXTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class IfVertexTest
{

private:

public:

                   IfVertexTest        ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
