from typing import Sequence, Dict

from symforce import symbolic as sf


class NamedVector:
    VARIABLES: Sequence[str] = ()

    def __init__(self, *args, **kwargs) -> None:

        if not args and not kwargs:
            self.matrix_ = sf.Matrix(*self.shape)
        else:
            self.matrix_ = sf.Matrix(*args, **kwargs)
            # TODO check shape

        self.variables_: Dict[str, int] = {}

        for i, name in enumerate(self.VARIABLES):
            self.variables_[name] = i

    @property
    def shape(self):
        return (len(self.VARIABLES), 1)

    def as_matrix(self):
        return self.matrix_

    def check_variable(self, key: str) -> bool:
        if key not in self.variables_:
            raise LookupError(f"{self.__class__.__name__} has no variable: '{key}'")

        return True

    def __getitem__(self, *args) -> sf.Matrix:
        # Single string
        if len(args) == 1 and isinstance(args[0], str):
            self.check_variable(args[0])
            return self.matrix_[self.variables_[args[0]]]

        # 1D list of strings
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            output = []

            for arg in args[0]:
                assert isinstance(arg, str)
                self.check_variable(arg)
                output.append(self.matrix_[self.variables_[arg]])

            return sf.Matrix(output)

        # Multiple strings
        if len(args) > 1:
            if all(isinstance(arg, str) for arg in args):
                output = []

                for arg in args:
                    self.check_variable(arg)
                    output.append(self.matrix_[self.variables_[arg]])

                return sf.Matrix(output)

        # Otherwise treat as normal sf.Matrix
        return self.matrix_[args]

    def __setitem__(self, key, value) -> None:
        if isinstance(key, str):
            self.check_variable(key)
            self.matrix_[self.variables_[key]] = value
            return

        if isinstance(key, (tuple, list)) and isinstance(value, (tuple, list)):
            assert len(key) == len(value)

            for k, v in zip(key, value):
                assert isinstance(k, str)
                self.check_variable(k)
                self.matrix_[self.variables_[k]] = v

            return
