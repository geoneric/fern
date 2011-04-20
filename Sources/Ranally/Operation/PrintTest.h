#ifndef INCLUDED_RANALLY_OPERATION_PRINTTEST
#define INCLUDED_RANALLY_OPERATION_PRINTTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class PrintTest
{

public:

                   PrintTest           ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

private:

};

#endif
