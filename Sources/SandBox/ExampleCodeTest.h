#ifndef INCLUDED_EXAMPLECODETEST
#define INCLUDED_EXAMPLECODETEST



// External headers.
#ifndef INCLUDED_VECTOR
#include <vector>
#define INCLUDED_VECTOR
#endif

// Project headers.

// Module headers.



namespace boost {
  namespace unit_test {
    class test_suite;
  }
}

namespace {
  // ExampleCodeTest declarations.
}



// namespace {

//! This class implements the unit tests for the ExampleCode class.
class ExampleCodeTest
{

private:

  std::vector<int> _intCollection1;

  std::vector<int> _intCollection2;

  std::vector<int> _intResult;

  std::vector<float> _floatCollection1;

  std::vector<float> _floatCollection2;

  std::vector<float> _floatResult;

  std::vector<bool> _boolResult;

  int              _intConstant;

  void             init                ();

public:

                   ExampleCodeTest     ();

  void             testConstructor     ();

  void             testCollectionCollection();

  void             testCollectionConstant();

  void             testMask            ();

  void             testSqrt            ();

  void             testAbs             ();

  void             testGreaterThan     ();

  void             testOverloads       ();

  void             testNAryOperation   ();

  void             testIsNull          ();

  void             testSetNull         ();

  void             testCollectorOperation();

  void             testIgnoreNoData    ();

  static boost::unit_test::test_suite* suite();

};

// } // namespace

#endif
