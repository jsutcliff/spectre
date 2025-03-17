from typing import Sequence, Dict, List

from symforce import symbolic as sf


class NamedVector:

    VARIABLES: Sequence[str] = ()

    def __init__(self, *args, **kwargs) -> None:

        # Create actual matrix storage
        if not args and not kwargs:
            self.matrix_ = sf.Matrix(self.length, 1)
        else:
            self.matrix_ = sf.Matrix(*args, **kwargs)

        # Populate variable lookup dict
        self.variables_: Dict[str, int] = {}

        for i, name in enumerate(self.VARIABLES):
            self.variables_[name] = i

        # Verify all is well
        if self.length != self.matrix_.SHAPE[0]:
            raise ValueError(f"{self.__class__.__name__} shape: {self.length} does not match sf.Matrix shape: {self.matrix_.SHAPE}")

    @property
    def length(self) -> int:
        """Number of variables in this NamedVector

        Returns:
            int: Number of variables
        """

        return len(self.VARIABLES)

    @property
    def empty(self) -> bool:
        """Checks if NamedVector has and variables set

        Returns:
            bool: True if length=0
        """

        return self.length > 0

    @property
    def variables(self) -> List[str]:
        """Gets variable names in NamedVector instance

        Returns:
            List[str]: List of varible names
        """

        return list(self.variables_.keys())

    def as_matrix(self) -> sf.Matrix:
        """Gets underlying Symforce Matrix

        Returns:
            sf.Matrix: Internal sf.Matrix
        """

        return self.matrix_

    @classmethod
    def as_symbolic_matrix(cls, prefix: str = "") -> sf.Matrix:
        """Classmethod that creates a sf.Matrix full of sf.Symbols with appropriate names

        Returns:
            sf.Matrix: Symforce Matrix containing symbols
        """

        symbols = [sf.Symbol(prefix + c) for c in cls.VARIABLES]
        return sf.Matrix(symbols)

    @classmethod
    def is_configured(cls) -> bool:
        """Checks if NamedVector has been configured with any variables

        Returns:
            bool: True if any variables are set
        """

        return len(cls.VARIABLES) > 0

    def _check_variable(self, key: str) -> bool:
        if key not in self.variables_:
            raise LookupError(f"{self.__class__.__name__} has no variable: '{key}'")

        return True

    def __getitem__(self, *args) -> sf.Matrix:
        # Single string
        if len(args) == 1 and isinstance(args[0], str):
            self._check_variable(args[0])
            return self.matrix_[self.variables_[args[0]]]

        # 1D list of strings
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            output = []

            for arg in args[0]:
                assert isinstance(arg, str)
                self._check_variable(arg)
                output.append(self.matrix_[self.variables_[arg]])

            return sf.Matrix(output)

        # Multiple strings
        if len(args) > 1:
            if all(isinstance(arg, str) for arg in args):
                output = []

                for arg in args:
                    self._check_variable(arg)
                    output.append(self.matrix_[self.variables_[arg]])

                return sf.Matrix(output)

        # Otherwise treat as normal sf.Matrix
        return self.matrix_[args]

    def __setitem__(self, key, value) -> None:
        # Single string item
        if isinstance(key, str):
            self._check_variable(key)
            self.matrix_[self.variables_[key]] = value
            return

        # List of strings and values
        if isinstance(key, (tuple, list)) and isinstance(value, (tuple, list)):
            if len(key) != len(value):
                raise ValueError(f"Number of variables: {len(key)} does not match number of values: {len(value)}")

            for k, v in zip(key, value):
                if not isinstance(k, str):
                    raise TypeError(f"Variable type expects str, got: {type(k)}")

                self._check_variable(k)
                self.matrix_[self.variables_[k]] = v

            return


class NamedMatrix(sf.Matrix):

    @classmethod
    def as_symbolic_matrix(cls, name: str = "mat"):
        """Classmethod that generates Symforce Matrix with symbolic names

        Args:
            name (str, optional): Symbolic naming prefix. Defaults to "mat".

        Returns:
            _type_: sf.Matrix full of symbols
        """
        return sf.Matrix([[sf.Symbol(f"{name}_{i+1}{j+1}") for j in range(cls.SHAPE[0])] for i in range(cls.SHAPE[1])])
