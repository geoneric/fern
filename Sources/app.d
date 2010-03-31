#!dmd -run

import std.stdio;

import ranally.operations.local.algorithms.binary.plus;
import ranally.operations.local.framework.binaryoperation;

// TODO How about references?



void main()
{
  writeln("Hello, world!");
  // writeln(double.max);
  // writeln(double.max + double.max);
  // writeln(double.min);
  // writeln(-double.min - double.min);
  // int result = ra.Plus.Plus!(int).algorithm(3, 4);
  // writeln("3 + 4");
  // writeln(result);
  // writeln(ra.Plus.Plus!(int).DomainPolicy.inDomain(3, 4));
  // writeln(int.max);     // 2147483647
  // writeln(int.max + 1); // -2147483648
}

unittest
{
  {
    BinaryOperation!(Plus!(int)) plus;
    int result = -9;
    plus(result, 4, 5);
    assert(result == 9);
  }

  {
    BinaryOperation!(Plus!(double)) plus;
    double result = -9.9;
    plus(result, 4.5, 5.5);
    assert(result == 10.0);
  }
}
