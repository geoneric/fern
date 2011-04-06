#ifndef INCLUDED_RANALLY_OPERATION_XMLPARSERTEST
#define INCLUDED_RANALLY_OPERATION_XMLPARSERTEST



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}



class XmlParserTest
{

public:

                   XmlParserTest       ();

  void             testParse           ();

  static boost::unit_test::test_suite* suite();

private:

};

#endif
