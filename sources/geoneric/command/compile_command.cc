#include "geoneric/command/compile_command.h"
#include "geoneric/core/exception.h"
#include "geoneric/ast/visitor/ast_dot_visitor.h"
#include "geoneric/ast/visitor/flowgraph_dot_visitor.h"
#include "geoneric/compiler/compiler.h"


namespace geoneric {
namespace {

void show_compile_help()
{
    std::cout <<
        "usage: geoneric compile [--help] LANGUAGE [ARGS]\n"
        "\n"
        "Compile the script to a target language.\n"
        "\n"
        "languages:\n"
        "  geoneric             Round-trip script\n"
        "  dot                 Compile script to Dot graph\n"
        "  c++                 Compile script to C++ code\n"
        "  python              Compile script to C++ code for Python extension\n"
        "  xml                 Compile script to XML\n"
        "\n"
        "See 'geoneric compile LANGUAGE --help' for more information on a specific\n"
        "language.\n"
        ;
}


void show_compile_cpp_help()
{
    std::cout <<
        "usage: geoneric compile cpp [--help] [--dump_driver]\n"
        "                            INPUT_SCRIPT OUTPUT_MODULE\n"
        "\n"
        "Compile the script module to a C++ module.\n"
        "\n"
        "  INPUT_SCRIPT        Script to compile or - to read from standard input\n"
        "  OUTPUT_MODULE       File to write result to, without an extension\n"
        ;
}


void show_compile_dot_help()
{
    std::cout <<
        "usage: geoneric compile dot [--help] GRAPH_TYPE [ARGS]\n"
        "\n"
        "Compile the script to a dot graph.\n"
        "\n"
        "graph types:\n"
        "  ast                 Abstract syntax tree\n"
        "  flowgraph           Flowgraph\n"
        "\n"
        "See 'geoneric compile dot GRAPH_TYPE --help' for more information on a\n"
        "specific graph type.\n"
        ;
}


void show_compile_dot_ast_help()
{
    std::cout <<
        "usage: geoneric compile dot ast [--help] [--with-cfg] [--with-use]\n"
        "                               INPUT_SCRIPT OUTPUT_SCRIPT\n"
        "\n"
        "Compile the script to a dot graph containing the abstract syntax tree.\n"
        "\n"
        "  INPUT_SCRIPT        Script to compile or - to read from standard input\n"
        "  OUTPUT_SCRIPT       File to write result to\n"
        "\n"
        "The result is written to standard output if no output script is provided\n"
        ;
}


void show_compile_dot_flowgraph_help()
{
    std::cout <<
        "usage: geoneric compile dot flowgraph [--help] INPUT_SCRIPT OUTPUT_SCRIPT\n"
        "\n"
        "Compile the script to a dot graph containing the flow graph.\n"
        "\n"
        "  INPUT_SCRIPT        Script to compile or - to read from standard input\n"
        "  OUTPUT_SCRIPT       File to write result to\n"
        "\n"
        "The result is written to standard output if no output script is provided\n"
        ;
}


// void show_compile_geoneric_help()
// {
//     std::cout <<
//         "usage: geoneric compile geoneric INPUT_SCRIPT [OUTPUT_SCRIPT]\n"
//         "\n"
//         "Compile the script to a geoneric script (round-trip).\n"
//         "\n"
//         "  INPUT_SCRIPT        Script to compile or - to read from standard input\n"
//         "  OUTPUT_SCRIPT       File to write result to\n"
//         "\n"
//         "The result is written to standard output if no output script is provided\n"
//         ;
// }


void show_compile_xml_help()
{
    std::cout <<
        "usage: geoneric compile xml [--help] INPUT_SCRIPT OUTPUT_SCRIPT\n"
        "\n"
        "Compile the script to xml.\n"
        "\n"
        "  INPUT_SCRIPT        Script to compile or - to read from standard input\n"
        "  OUTPUT_SCRIPT       File to write result to\n"
        "\n"
        "The result is written to standard output if no output script is provided\n"
        ;
}

} // Anonymous namespace


CompileCommand::CompileCommand(
    int argc,
    char** argv)

    : Command(argc, argv)

{
}


int CompileCommand::compile_to_geoneric(
    int /* argc */,
    char** /* argv */) const
{
    std::cout << "Conversion to Geoneric script not supported yet\n";
    return EXIT_SUCCESS;
}


int CompileCommand::compile_to_cpp(
    int argc,
    char** argv) const
{
    int status = EXIT_FAILURE;

    if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
        // No arguments, or the help option.
        show_compile_cpp_help();
        status = EXIT_SUCCESS;
    }
    else {
        int current_argument_id = 1;
        Compiler::Flags flags = 0;

        while(current_argument_id < argc) {
            if(std::strcmp(argv[current_argument_id], "--dump_driver") == 0) {
                flags |= Compiler::DUMP_DRIVER;
                ++current_argument_id;
            }
            else if(std::strcmp(argv[current_argument_id], "--dump_cmake") ==
                    0) {
                flags |= Compiler::DUMP_CMAKE;
                ++current_argument_id;
            }
            else {
                break;
            }
        }

        if(argc - current_argument_id < 2) {
            std::cerr << "Not enough arguments.\n";
            show_compile_cpp_help();
            status = EXIT_FAILURE;
        }
        else if(argc - current_argument_id > 2) {
            std::cerr << "Too many arguments.\n";
            show_compile_cpp_help();
            status = EXIT_FAILURE;
        }
        else {
            Path input_file_path =
                std::strcmp(argv[current_argument_id], "-") != 0
                    ? argv[current_argument_id] : "";
            ++current_argument_id;
            Path output_directory_path = argv[current_argument_id];
            Compiler compiler("h", "cc");
            compiler.compile(input_file_path, output_directory_path, flags);
            status = EXIT_SUCCESS;
        }
    }

    return status;
}


String CompileCommand::compile_to_dot_ast(
    std::shared_ptr<ModuleVertex> const& tree,
    int modes) const
{
    const_cast<Interpreter&>(interpreter()).annotate(tree);

    AstDotVisitor ast_dot_visitor(modes);
    tree->Accept(ast_dot_visitor);

    return ast_dot_visitor.script();
}


int CompileCommand::compile_to_dot_ast(
    int argc,
    char** argv) const
{
    int status = EXIT_FAILURE;

    if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
        // No arguments, or the help option.
        show_compile_dot_ast_help();
        status = EXIT_SUCCESS;
    }
    else {
        int current_argument_id = 1;
        int modes = 0x0;
        while(current_argument_id < argc) {
            if(std::strcmp(argv[current_argument_id], "--with-cfg") == 0) {
                modes |= AstDotVisitor::ConnectingCfg;
                ++current_argument_id;
            }
          else if(std::strcmp(argv[current_argument_id], "--with-use") == 0) {
              modes |= AstDotVisitor::ConnectingUses;
              ++current_argument_id;
          }
          else {
              break;
          }
        }

        if(current_argument_id == argc) {
            std::cerr << "Not enough arguments.\n";
            show_compile_dot_ast_help();
            status = EXIT_FAILURE;
        }
        else if(argc - current_argument_id > 3) {
            std::cerr << "Too many arguments.\n";
            show_compile_dot_ast_help();
            status = EXIT_FAILURE;
        }
        else {
            std::string input_filename =
                std::strcmp(argv[current_argument_id], "-") != 0
                    ? argv[current_argument_id] : "";
            ++current_argument_id;
            std::string output_filename = current_argument_id == argc - 1
                ? argv[current_argument_id] : "";
            ModuleVertexPtr tree(interpreter().parse_file(input_filename));
            String dot_script = compile_to_dot_ast(tree, modes);
            write(dot_script, output_filename);
            status = EXIT_SUCCESS;
        }
    }

    return status;
}


String CompileCommand::compile_to_dot_flowgraph(
    ModuleVertexPtr const& tree) const
{
    const_cast<Interpreter&>(interpreter()).annotate(tree);

    FlowgraphDotVisitor flowgraph_dot_visitor;
    tree->Accept(flowgraph_dot_visitor);

    return flowgraph_dot_visitor.script();
}


int CompileCommand::compile_to_dot_flowgraph(
    int argc,
    char** argv) const
{
    int status = EXIT_FAILURE;

    if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
        // No arguments, or the help option.
        show_compile_dot_flowgraph_help();
        status = EXIT_SUCCESS;
    }
    else {
        int current_argument_id = 1;

        if(current_argument_id == argc) {
            std::cerr << "Not enough arguments.\n";
            show_compile_dot_ast_help();
            status = EXIT_FAILURE;
        }
        else if(argc - current_argument_id > 3) {
            std::cerr << "Too many arguments.\n";
            show_compile_dot_ast_help();
            status = EXIT_FAILURE;
        }
        else {
            std::string input_filename =
                std::strcmp(argv[current_argument_id], "-") != 0
                    ? argv[current_argument_id] : "";
            ++current_argument_id;
            std::string output_filename = current_argument_id == argc - 1
                ? argv[current_argument_id] : "";
            ModuleVertexPtr tree(interpreter().parse_file(input_filename));
            String dot_script = compile_to_dot_flowgraph(tree);
            write(dot_script, output_filename);
            status = EXIT_SUCCESS;
        }
    }

    return status;
}


int CompileCommand::compile_to_dot(
    int argc,
    char** argv) const
{
    int status = EXIT_FAILURE;

    if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
        // No arguments, or the help option.
        show_compile_dot_help();
        status = EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[1], "ast") == 0) {
        status = compile_to_dot_ast(argc - 1, argv + 1);
    }
    else if(std::strcmp(argv[1], "flowgraph") == 0) {
        status = compile_to_dot_flowgraph(argc - 1, argv + 1);
    }
    else {
        std::cerr << "Unknown graph type: " << argv[1] << "\n";
        std::cerr << "See 'geoneric compile dot --help' for list of types.\n";
        status = EXIT_FAILURE;
    }

    return status;
}


int CompileCommand::compile_to_python(
    int /* argc */,
    char** /* argv */) const
{
    std::cout << "Conversion to Python not supported yet\n";
    return EXIT_SUCCESS;
}


int CompileCommand::compile_to_xml(
    int argc,
    char** argv) const
{
    int status = EXIT_FAILURE;

    if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
        // No arguments, or the help option.
        show_compile_xml_help();
        status = EXIT_SUCCESS;
    }
    else {
        int current_argument_id = 1;

        if(argc - current_argument_id > 2) {
            std::cerr << "Too many arguments.\n";
            show_compile_xml_help();
            status = EXIT_FAILURE;
        }
        else {
            std::string input_filename =
                std::strcmp(argv[current_argument_id], "-") != 0
                    ? argv[current_argument_id] : "";
            ++current_argument_id;
            std::string output_filename = current_argument_id == argc - 1
                ? argv[current_argument_id] : "";
            // String xml = read(input_filename);
            // write(xml, output_filename);
            std::cout << "TODO make XML visitor" << std::endl;
            status = EXIT_SUCCESS;
        }
    }

    return status;
}


int CompileCommand::execute() const
{
    int status = EXIT_FAILURE;

    try {
        if(argc() == 1 || std::strcmp(argv()[1], "--help") == 0) {
            // No arguments, or the help option.
            show_compile_help();
            status = EXIT_SUCCESS;
        }
        else if(std::strcmp(argv()[1], "geoneric") == 0) {
            status = compile_to_geoneric(argc() - 1, argv() + 1);
        }
        else if(std::strcmp(argv()[1], "dot") == 0) {
            status = compile_to_dot(argc() - 1, argv() + 1);
        }
        else if(std::strcmp(argv()[1], "c++") == 0) {
            status = compile_to_cpp(argc() - 1, argv() + 1);
        }
        else if(std::strcmp(argv()[1], "python") == 0) {
            status = compile_to_python(argc() - 1, argv() + 1);
        }
        else if(std::strcmp(argv()[1], "xml") == 0) {
            status = compile_to_xml(argc() - 1, argv() + 1);
        }
        else {
            std::cerr << "Unknown target language: " << argv()[1] << "\n";
            std::cerr << "See 'geoneric compile --help' for list of languages.\n";
            status = EXIT_FAILURE;
        }
    }
    catch(Exception const& exception) {
        std::cerr << exception.message() << "\n";
        status = EXIT_FAILURE;
    }
    catch(std::exception const& exception) {
        std::cerr << exception.what() << "\n";
        status = EXIT_FAILURE;
    }

    return status;
}

} // namespace geoneric
