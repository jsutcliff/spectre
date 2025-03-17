import os
from typing import Callable, List

from symforce.codegen import Codegen, CppConfig, PythonConfig


def generate_cpp_function(function_name: Callable, output_names: List[str], output_dir: str = "generated"):

    codegen = Codegen.function(function_name, output_names=output_names, config=CppConfig())

    metadata = codegen.generate_function(output_dir=output_dir, skip_directory_nesting=True)

    for f in metadata.generated_files:
        print("  |- {}".format(os.path.relpath(f, metadata.output_dir)))

    # TODO
    # if (replace_unused_vars):
