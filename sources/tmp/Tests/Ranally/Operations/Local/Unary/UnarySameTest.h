#ifndef INCLUDED_UNARYSAMETEST
#define INCLUDED_UNARYSAMETEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class UnarySameTest
{

private:

public:

                   UnarySameTest       ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

};

#endif
