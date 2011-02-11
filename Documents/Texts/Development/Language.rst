********
Language
********

Requirements
============
* Operations are named after their purpose, not after the algorithms used to implement them.
* Operations may have arguments with which the operation can be tweaked, if relevant. For example, alternative implementations (algorithms) can be selected, that may have different accuracy characteristics.

Scripting language
==================
The Python scripting language syntax is used as the textual user interface for the user. Internally, the folowing conversions are being performed:

1. Script is parsed using Python.
2. Python's internal syntax tree is converted to XML.
3. XML is parsed and immediately converted to a syntax tree.

XML is used as an intermediate format because it can contain more information than the script language can. Clients that don't need/use the scripting language, are able to pass additional information about the task in XML.

Information about the CPython's syntax tree can be found here: http://www.python.org/dev/peps/pep-0339/. Relevant header file is `Python-ast.h`.


