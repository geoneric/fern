#include "fern/compiler/compiler.h"
#include "fern/io/core/file.h"
#include "fern/compiler/compile_visitor.h"


namespace fern {

//! Constructor.
/*!
  \param     header_extension Filename extension to use for header files.
  \param     module_extension Filename extension to use for module files.
*/
Compiler::Compiler(
    String const& header_extension,
    String const& module_extension)

    : _interpreter(),
      _header_extension(header_extension),
      _module_extension(module_extension)


{
    assert(!_header_extension.is_empty());
    assert(!_module_extension.is_empty());
    assert(_header_extension != _module_extension);
    assert(!_header_extension.starts_with("."));
    assert(!_module_extension.starts_with("."));
}


//! Compile source module to C++ code and write a destination module.
/*!
  \param     source_module_path Path to source module.
  \param     destination_module_path Path to destination module, without
             extension. Two files will be written: a header file and an
             module file.
  \return    .
  \exception .
  \warning   .
  \sa        .
*/
void Compiler::compile(
    Path const& source_module_path,
    Path const& destination_module_path,
    Flags flags)
{
    // Parse module into a syntax tree.
    // TODO Once it is possible to parse modules, make sure not to recurse into
    //      modules here.
    ModuleVertexPtr module_vertex = _interpreter.parse_file(source_module_path);
    assert(module_vertex);

    // TODO Exception.
    assert(directory_is_writable(destination_module_path));

    String model_name = Path(module_vertex->source_name()).filename().stem();
    Path header_path = destination_module_path /
        (model_name + "." + _header_extension);
    Path module_path = destination_module_path /
        (model_name + "." + _module_extension);

    {
        bool can_overwrite_existing_file = file_exists(header_path) &&
            file_is_writable(header_path);
        bool can_create_unexisting_file = (!file_exists(header_path)) &&
            directory_is_writable(destination_module_path.parent_path());

        if(!(can_overwrite_existing_file || can_create_unexisting_file)) {
            // TODO Exception.
            assert(false);
        }
    }

    {
        bool can_overwrite_existing_file = file_exists(module_path) &&
            file_is_writable(module_path);
        bool can_create_unexisting_file = (!file_exists(module_path)) &&
            directory_is_writable(destination_module_path.parent_path());

        if(!(can_overwrite_existing_file || can_create_unexisting_file)) {
            // TODO Exception.
            assert(false);
        }
    }

    CompileVisitor visitor(header_path.filename());
    module_vertex->Accept(visitor);

    write_file(visitor.header(), header_path);
    write_file(visitor.module(), module_path);

    if(flags & DUMP_CMAKE) {
        Path cmake_file_path = destination_module_path / "CMakeLists.txt";
        String project_name = model_name;

        String cmake_script;
        std::vector<String> installable_targets;

        cmake_script += String((boost::format(
            "CMAKE_MINIMUM_REQUIRED(VERSION 2.8)\n"
            "PROJECT(%1%)\n"
            "SET(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})\n"
            "INCLUDE(FernCompiler)\n"
            "INCLUDE(FernExternal)\n"
        ) % project_name).str());


        String library_target = model_name + "_library";
        {
            installable_targets.push_back(library_target);
            std::vector<String> library_module_names({
                model_name
            });
            std::vector<String> link_libraries({
            });

            cmake_script += String((boost::format(
                "ADD_LIBRARY(%1%\n"
                "    %2%\n"
                ")\n"
                "SET_TARGET_PROPERTIES(%1%\n"
                "    PROPERTIES\n"
                "        OUTPUT_NAME %3%\n"
                ")\n"
                "TARGET_LINK_LIBRARIES(%1%\n"
                "    %4%\n"
                ")\n"
            ) % library_target % join(library_module_names, ", ")
                % model_name % join(link_libraries, " ")).str());
        }

        if(flags & DUMP_DRIVER) {
            String driver_target = model_name + "_driver";
            installable_targets.push_back(driver_target);
            std::vector<String> driver_module_names({
                "main"
            });
            std::vector<String> link_libraries({
                library_target,
                "fernlib",
                "${FERN_EXTERNAL_LIBRARIES}"
            });
            cmake_script += String((boost::format(
                "ADD_EXECUTABLE(%1%\n"
                "    %2%\n"
                ")\n"
                "SET_TARGET_PROPERTIES(%1%\n"
                "    PROPERTIES\n"
                "        OUTPUT_NAME %3%\n"
                ")\n"
                "TARGET_LINK_LIBRARIES(%1%\n"
                "    %4%\n"
                ")\n"
            ) % driver_target % join(driver_module_names, ", ")
                % model_name % join(link_libraries, " ")).str());
        }

        cmake_script += String((boost::format(
            "INSTALL(\n"
            "    TARGETS %1%\n"
            "    RUNTIME DESTINATION bin\n"
            "    ARCHIVE DESTINATION lib\n"
            "    LIBRARY DESTINATION lib\n"
            ")\n"
        ) % join(installable_targets, " ")).str());

        write_file(cmake_script, cmake_file_path);
    }

    if(flags & DUMP_DRIVER) {
        Path driver_module_path = (destination_module_path / "main")
            .replace_extension(_module_extension);
        String class_name = model_name;
        write_file(String((boost::format(
            "#include <cstdlib>\n"
            "#include <iostream>\n"
            "#include \"%1%\"\n"
            "\n"
            "\n"
            "int main(\n"
            "        int argc,\n"
            "        char** argv)\n"
            "{\n"
            "    int status = EXIT_FAILURE;\n"
            "\n"
            "    try {\n"
            "        %2% model(argc, argv);\n"
            "        model.run();\n"
            "        status = EXIT_SUCCESS;\n"
            "    }\n"
            "    catch(std::exception const& exception) {\n"
            "        std::cerr << exception.what() << std::endl;\n"
            "    }\n"
            "\n"
            "    return status;\n"
            "}"
        ) % header_path.filename() % class_name).str()), driver_module_path);
    }
}

} // namespace fern
