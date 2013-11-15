*********************
Commandline utilities
*********************

fern
====
Main command to unlock the functionality from the commandline.

.. code-block:: c++

  // TODO Think of all tasks this utility needs to perform and design a
  //      command line interface for that.
  //      - interpret/execute script
  //      - convert script to script (roundtrip)
  //      - convert script to dot-graph
  //      - convert script to C++
  //      - convert script to C++, including Python extension
  // Some of these actions can happen simultaneously. Others are silly to
  // combine.
  //   valid: all conversions
  //   silly: interpret and convert

  // Ideas:
  // - Execute is default/implied if no conversion action is provided.
  // - Convert argument has a value of script | dot | C++ | Python
  // - Iff convert argument is provided, than no execution takes place.
  //   Positional arguments are related to the conversion process
  //   (target dir / file, ...).
  //
  // - execute command is implied.
  // - first positional is filename of script.
  // - if --script option is provided, than script positional should not be.
  //   fern model.ran
  //   fern execute model.ran
  //   fern --script "slope = slope("dem")"
  //   fern execute --script "slope = slope("dem")"
  //
  // - first positional is language argument.
  // - 
  //   fern convert dot model.ran model.dot
  //   fern convert c++ model.ran
  //   fern convert c++ --namespace bla --output-directory ./blo model.ran
  //   fern convert python --package bli model.ran
  //
  //   The Python extension converter should not create the core C++ code.
  //   The c++ converter is for doing that. Or can we use two convert commands?

