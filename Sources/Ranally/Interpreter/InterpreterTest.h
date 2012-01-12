#ifndef INCLUDED_RANALLY_INTERPRETER_INTERPRETERTEST
#define INCLUDED_RANALLY_INTERPRETER_INTERPRETERTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class InterpreterTest
{

public:

                   InterpreterTest     ();

  void             test                ();

  static boost::unit_test::test_suite* suite();

private:

};

#endif
