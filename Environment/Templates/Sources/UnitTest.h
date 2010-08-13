#ifndef INCLUDED_RANALLY_TESTCLASS
#define INCLUDED_RANALLY_TESTCLASS



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class ClassTest
{

private:

public:

                   ClassTest           ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
