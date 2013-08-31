#include "ranally/command/convert_command.h"
#include "ranally/core/exception.h"
#include "ranally/ast/visitor/ast_dot_visitor.h"
#include "ranally/ast/visitor/flowgraph_dot_visitor.h"


namespace ranally {
namespace {

void show_convert_help()
{
    std::cout <<
        "usage: ranally convert [--help] LANGUAGE [ARGS]\n"
        "\n"
        "Convert the script to a target language.\n"
        "\n"
        "languages:\n"
        "  ranally             Round-trip script\n"
        "  dot                 Convert script to Dot graph\n"
        "  c++                 Convert script to C++ code\n"
        "  python              Convert script to C++ code for Python extension\n"
        "  xml                 Convert script to XML\n"
        "\n"
        "See 'ranally convert LANGUAGE --help' for more information on a specific\n"
        "language.\n"
        ;
}


void show_convert_dot_help()
{
    std::cout <<
        "usage: ranally convert dot [--help] GRAPH_TYPE [ARGS]\n"
        "\n"
        "Convert the script to a dot graph.\n"
        "\n"
        "graph types:\n"
        "  ast                 Abstract syntax tree\n"
        "  flowgraph           Flowgraph\n"
        "\n"
        "See 'ranally convert dot GRAPH_TYPE --help' for more information on a\n"
        "specific graph type.\n"
        ;
}


void show_convert_dot_ast_help()
{
    std::cout <<
        "usage: ranally convert dot ast [--help] [--with-cfg] [--with-use]\n"
        "                               INPUT_SCRIPT OUTPUT_SCRIPT\n"
        "\n"
        "Convert the script to a dot graph containing the abstract syntax tree.\n"
        "\n"
        "  INPUT_SCRIPT        Script to convert or - to read from standard input\n"
        "  OUTPUT_SCRIPT       File to write result to\n"
        "\n"
        "The result is written to standard output if no output script is provided\n"
        ;
}


void show_convert_dot_flowgraph_help()
{
    std::cout <<
        "usage: ranally convert dot flowgraph [--help] INPUT_SCRIPT OUTPUT_SCRIPT\n"
        "\n"
        "Convert the script to a dot graph containing the flow graph.\n"
        "\n"
        "  INPUT_SCRIPT        Script to convert or - to read from standard input\n"
        "  OUTPUT_SCRIPT       File to write result to\n"
        "\n"
        "The result is written to standard output if no output script is provided\n"
        ;
}


// void show_convert_ranally_help()
// {
//     std::cout <<
//         "usage: ranally convert ranally INPUT_SCRIPT [OUTPUT_SCRIPT]\n"
//         "\n"
//         "Convert the script to a ranally script (round-trip).\n"
//         "\n"
//         "  INPUT_SCRIPT        Script to convert or - to read from standard input\n"
//         "  OUTPUT_SCRIPT       File to write result to\n"
//         "\n"
//         "The result is written to standard output if no output script is provided\n"
//         ;
// }


void show_convert_xml_help()
{
    std::cout <<
        "usage: ranally convert xml [--help] INPUT_SCRIPT OUTPUT_SCRIPT\n"
        "\n"
        "Convert the script to xml.\n"
        "\n"
        "  INPUT_SCRIPT        Script to convert or - to read from standard input\n"
        "  OUTPUT_SCRIPT       File to write result to\n"
        "\n"
        "The result is written to standard output if no output script is provided\n"
        ;
}

} // Anonymous namespace


ConvertCommand::ConvertCommand(
    int argc,
    char** argv)

    : Command(argc, argv)

{
}


int ConvertCommand::convert_to_ranally(
    int /* argc */,
    char** /* argv */) const
{
    std::cout << "Conversion to Ranally script not supported yet\n";
    return EXIT_SUCCESS;
}


int ConvertCommand::convert_to_cpp(
    int /* argc */,
    char** /* argv */) const
{
    std::cout << "Conversion to C++ not supported yet\n";
    return EXIT_SUCCESS;
}


String ConvertCommand::convert_to_dot_ast(
    std::shared_ptr<ModuleVertex> const& tree,
    int modes) const
{
    const_cast<Interpreter&>(interpreter()).annotate(tree);

    AstDotVisitor ast_dot_visitor(modes);
    tree->Accept(ast_dot_visitor);

    return ast_dot_visitor.script();
}


int ConvertCommand::convert_to_dot_ast(
    int argc,
    char** argv) const
{
    int status = EXIT_FAILURE;

    if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
        // No arguments, or the help option.
        show_convert_dot_ast_help();
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
            show_convert_dot_ast_help();
            status = EXIT_FAILURE;
        }
        else if(argc - current_argument_id > 3) {
            std::cerr << "Too many arguments.\n";
            show_convert_dot_ast_help();
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
            String dot_script = convert_to_dot_ast(tree, modes);
            write(dot_script, output_filename);
            status = EXIT_SUCCESS;
        }
    }

    return status;
}


String ConvertCommand::convert_to_dot_flowgraph(
    ModuleVertexPtr const& tree) const
{
    const_cast<Interpreter&>(interpreter()).annotate(tree);

    FlowgraphDotVisitor flowgraph_dot_visitor;
    tree->Accept(flowgraph_dot_visitor);

    return flowgraph_dot_visitor.script();
}


int ConvertCommand::convert_to_dot_flowgraph(
    int argc,
    char** argv) const
{
    int status = EXIT_FAILURE;

    if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
        // No arguments, or the help option.
        show_convert_dot_flowgraph_help();
        status = EXIT_SUCCESS;
    }
    else {
        int current_argument_id = 1;

        if(current_argument_id == argc) {
            std::cerr << "Not enough arguments.\n";
            show_convert_dot_ast_help();
            status = EXIT_FAILURE;
        }
        else if(argc - current_argument_id > 3) {
            std::cerr << "Too many arguments.\n";
            show_convert_dot_ast_help();
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
            String dot_script = convert_to_dot_flowgraph(tree);
            write(dot_script, output_filename);
            status = EXIT_SUCCESS;
        }
    }

    return status;
}


int ConvertCommand::convert_to_dot(
    int argc,
    char** argv) const
{
    int status = EXIT_FAILURE;

    if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
        // No arguments, or the help option.
        show_convert_dot_help();
        status = EXIT_SUCCESS;
    }
    else if(std::strcmp(argv[1], "ast") == 0) {
        status = convert_to_dot_ast(argc - 1, argv + 1);
    }
    else if(std::strcmp(argv[1], "flowgraph") == 0) {
        status = convert_to_dot_flowgraph(argc - 1, argv + 1);
    }
    else {
        std::cerr << "Unknown graph type: " << argv[1] << "\n";
        std::cerr << "See 'ranally convert dot --help' for list of types.\n";
        status = EXIT_FAILURE;
    }

    return status;
}


int ConvertCommand::convert_to_python(
    int /* argc */,
    char** /* argv */) const
{
    std::cout << "Conversion to Python not supported yet\n";
    return EXIT_SUCCESS;
}


int ConvertCommand::convert_to_xml(
    int argc,
    char** argv) const
{
    int status = EXIT_FAILURE;

    if(argc == 1 || std::strcmp(argv[1], "--help") == 0) {
        // No arguments, or the help option.
        show_convert_xml_help();
        status = EXIT_SUCCESS;
    }
    else {
        int current_argument_id = 1;

        if(argc - current_argument_id > 2) {
            std::cerr << "Too many arguments.\n";
            show_convert_xml_help();
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


int ConvertCommand::execute() const
{
    int status = EXIT_FAILURE;

    try {
        if(argc() == 1 || std::strcmp(argv()[1], "--help") == 0) {
            // No arguments, or the help option.
            show_convert_help();
            status = EXIT_SUCCESS;
        }
        else if(std::strcmp(argv()[1], "ranally") == 0) {
            status = convert_to_ranally(argc() - 1, argv() + 1);
        }
        else if(std::strcmp(argv()[1], "dot") == 0) {
            status = convert_to_dot(argc() - 1, argv() + 1);
        }
        else if(std::strcmp(argv()[1], "c++") == 0) {
            status = convert_to_cpp(argc() - 1, argv() + 1);
        }
        else if(std::strcmp(argv()[1], "python") == 0) {
            status = convert_to_python(argc() - 1, argv() + 1);
        }
        else if(std::strcmp(argv()[1], "xml") == 0) {
            status = convert_to_xml(argc() - 1, argv() + 1);
        }
        else {
            std::cerr << "Unknown target language: " << argv()[1] << "\n";
            std::cerr << "See 'ranally convert --help' for list of languages.\n";
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

} // namespace ranally
