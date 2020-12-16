from typing import List, Dict


class Compartment:
    """
    A compartment in the compartmental model.
    Each compartment is full of people and can be stratified into many smaller compartments.
    """

    def __init__(
        self,
        name: str,
        strat_names: List[str] = [],
        strat_values: Dict[str, str] = {},
    ):
        assert set(strat_names) == set(strat_values.keys()), "Mismatch in Compartment strat values."
        assert type(name) is str, "Name must be a string, not %s." % type(name)
        # Internal values, must not be modified by external code.
        self._name = name
        self._strat_names = tuple(strat_names)
        self._strat_values = strat_values
        self._str = self.serialize()
        self.idx = None

    def is_match(self, name: str, strata: dict) -> bool:
        """
        Returns True if this compartment matches the supplied name and strata.
        A partial strata match returns True.
        """
        is_name_match = name == self._name
        is_strata_match = all([self.has_stratum(k, v) for k, v in strata.items()])
        return is_name_match and is_strata_match

    def has_name(self, comp):
        """
        Returns True if this compartment has the same root name as another.
        """
        if type(comp) is str:
            return self._name == comp
        else:
            return self._name == comp._name

    def has_name_in_list(self, comps: list):
        """
        Returns True if this compartment has the same root name
        as any compartment in the provided list.
        """
        return any(self.has_name(c) for c in comps)

    def has_stratum(self, name: str, value: str):
        return self._strat_values.get(name) == value

    def get_strata(self) -> List[str]:
        return [self._serialize_stratum(s) for s in self._strat_names]

    def serialize(self) -> str:
        """Return string representation of the compartment"""
        return "X".join([self._name, *self.get_strata()])

    def stratify(self, stratify_name: str, stratum_name: str):
        new_strat_names = [*self._strat_names, stratify_name]
        new_strat_values = {**self._strat_values, stratify_name: stratum_name}
        return Compartment(
            name=self._name,
            strat_names=new_strat_names,
            strat_values=new_strat_values,
        )

    @staticmethod
    def deserialize(s: str):
        """Convert string into Compartment"""
        parts = s.split("X")
        name = parts[0]
        strat_names = []
        strat_values = {}
        for strat in parts[1:]:
            s_name, s_value = Compartment._deserialize_stratum(strat)
            strat_names.append(s_name)
            strat_values[s_name] = s_value

        return Compartment(name, strat_names, strat_values)

    @staticmethod
    def _deserialize_stratum(stratum_string):
        stratum_parts = stratum_string.split("_")
        name = stratum_parts[0]
        value = "_".join(stratum_parts[1:])
        return name, value

    def _serialize_stratum(self, stratum_name):
        return f"{stratum_name}_{self._strat_values[stratum_name]}"

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return self._str

    def __eq__(self, obj):
        """
        Equality operator override
        """
        is_same_as_str = type(obj) is str and obj == self._str
        is_same_as_comp = type(obj) is Compartment and obj._str == self._str
        return is_same_as_str or is_same_as_comp

    def __hash__(self):
        """
        Hash operator override
        """
        return hash(self._str)
