User stories
============

Environmental modelers
----------------------
   "As an environmental modeler, I want to still be able to run the models I create today 10 or more years from now."

Implications and tasks:

* Implemented using technology that has a good change of still being available after 10 or more years from now.
* Models should be written in a language that is fit for representing environmental models. This language has a formal specification that is publicly available (ofcourse) and allows other people to implement it (compare with C++, Python, XSLT, ...).
* It should be possible to translate models to other languages (C++, Python, Java, ...) so people can integrate them with other software. This is indirection and makes it possible to support the next popular language. Compare with xsd which can be used to translate XML Schema documents to C++ parsers.

..

   "As an environmental modeler, I want to be able to focus on the model rules and not be distracted by unnecessary technicallities."

Implications and tasks:

* Design a high level modeling language.
* Implement an interpreter.

..

   "As an environmental modeler, I want to be able to use uncertain spatio-temporal inputs."

Implications and tasks:

* Implicit spatial iteration in case one of the inputs is spatial.
* Implicit temporal iteration in case one of the inputs is temporal.
* Implicit Monte-Carlo simulation in case on of the inputs is uncertain.

..

   "As an environmental modeler, I want to be able to use observation data to calibrate my uncertain model inputs."

Implications and tasks:

* Implicit partical filtering in case one of the outputs is known at certain moments in time. These values are taken to be inputs to the particle filtering procedure.

..

   "As an environmental modeler, I want my model to be executed using all available hardware that I have."

Implications and tasks:

* Concurrent tasks must be distributed over the CPU cores and GPU cores.

Geo-ICT consultants
-------------------
   "As a geo-ICT software consultant, I want it to be easy to add support for environmental models to applications I write for my customers."

Implications and tasks:

* Express models in a high level scripting language.
* Implement a compiler which translates the model to some lower level programming language, like C++, Python, Java.

..

   "As a geo-ICT consultant, I want to be able to add operations to the modeling environment."

Implications and tasks:

* We must use a plugin system that makes it possible to dynamically load shared libraries containing additional operations. The standard library with operations must be loaded using this mechanism also.
* We need a namespace machanism that prevents multiple operations with the same name to clash. For example, the standard library has namespace std, some developer's library can be something like cool_stuff.
* The standard library is implicitly loaded(?).
* The symbols from the standard library are available without namespace qualification(?).

