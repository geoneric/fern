#ifndef INCLUDED_PLUSTEST
#define INCLUDED_PLUSTEST



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

  void             testDomain          ();

  void             testAlgorithm       ();

  void             testRange           ();

  static boost::unit_test::test_suite* suite();

};

#endif
