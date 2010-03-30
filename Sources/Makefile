# For the unit tests to be run, the file must be explicitly compiled. This is
# not necessary for the app to be build.
#
# ? Can compile time assertions be put in template scope?
#
# /*
#   Authors: Kor de Jong, kor@jemig.eu
#   Examples:
#   Returns:
#   See_Also:
#   Throws:
#   Params:
# */

sources := \
  app.d \
  ranally/operations/local/algorithms/binary/*.d \
  ranally/operations/local/policies/*.d

all:
	dmd -unittest ${sources}
	app

doc:
	dmd -Ddhtml ${sources}

clean:
	rm -f ${sources:.d=.o}
