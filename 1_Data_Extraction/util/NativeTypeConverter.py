from typing import Any
from pydicom.valuerep import DSfloat, IS, PersonName
from pydicom.multival import MultiValue
from pydicom.uid import UID


def convert_uid_to_native_type(uid: UID) -> str:
    return str(uid)


def convert_person_name_to_native_type(person_name: PersonName) -> str:
    return str(person_name)


def convert_is_to_native_type(is_: IS) -> int:
    return int(is_)


def convert_dsfloat_to_native_type(dsfloat_: DSfloat) -> float:
    return float(dsfloat_)


def convert_multivalue_to_native_type(multivalue: MultiValue) -> list:
    mv = list()
    # convert all values to native type:
    for i in range(len(multivalue)):
        mv.append(convert_to_native_type(multivalue[i]))
    return mv


def convert_to_native_type(value) -> Any:
    t = type(value)

    if t == MultiValue:
        return convert_multivalue_to_native_type(value)
    elif t == UID:
        return convert_uid_to_native_type(value)
    elif t == PersonName:
        return convert_person_name_to_native_type(value)
    elif t == IS:
        return convert_is_to_native_type(value)
    elif t == DSfloat:
        return convert_dsfloat_to_native_type(value)
    else:
        return value
