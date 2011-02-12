#ifndef INCLUDED_RANALLY_ALGORITHM_PLUSTEST
#define INCLUDED_RANALLY_ALGORITHM_PLUSTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class PlusTest
{

private:

public:

                   PlusTest            ();

  void             testArgumentAndResultTypes();

  static boost::unit_test::test_suite* suite();

};

#endif
