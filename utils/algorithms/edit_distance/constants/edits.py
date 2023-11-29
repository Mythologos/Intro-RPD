from aenum import NamedConstant


class EditOperation(NamedConstant):
    INSERT = 0
    DELETE = 1
    SUBSTITUTE = 2
