#!dmd -run

import std.stdio;

import ranally.operations.local.algorithms.binary.plus;



void main()
{
  writeln("Hello, world!");
  // int result = ra.Plus.Plus!(int).algorithm(3, 4);
  // writeln("3 + 4");
  // writeln(result);
  // writeln(ra.Plus.Plus!(int).DomainPolicy.inDomain(3, 4));
  // writeln(int.max);     // 2147483647
  // writeln(int.max + 1); // -2147483648
}

unittest
{
  assert(Plus!(int).algorithm(3, 4) == 7);
}
