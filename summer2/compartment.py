"""
This module contains a class used to represent compartments.
"""
from typing import Optional, Dict


class Compartment:
    """
    A single compartment in the compartmental model.
    Each compartment is defined by its name and strata.
    A compartment does not store the number of occupants - this data is tracked elsewhere in ``CompartmentalModel``.

    Args:
        name: The compartment's name
        strata: A mapping which defines this compartment's strata for each stratification.

    Example:
        Create a Compartment with no stratifications::

            comp = Compartment("susceptible")

        Create a Compartment with age and location stratifications::

            comp = Compartment(
                name="susceptible"
                strata={
                    "age": "young",
                    "location": "rural",
                }
            )

    """

    def __init__(
        self,
        name: str,
        strata: Optional[Dict[str, str]] = None,
    ):
        assert type(name) is str, "Name must be a string, not %s." % type(name)
        self.name = name
        self.strata = strata or {}
        self._str = self.serialize()
        self.idx = None

    def is_match(self, name: str, strata: dict) -> bool:
        """
        Determines whether this compartment matches the supplied name and strata.
        A partial strata match, where this compartment has more strata than are supplied returns True.
        """
        return name == self.name and self.has_strata(strata)

    def has_strata(self, strata: dict) -> bool:
        return all([self.has_stratum(k, v) for k, v in strata.items()])

    def has_stratum(self, name: str, value: str) -> bool:
        return self.strata.get(name) == value

    def has_name(self, comp) -> bool:
        """
        Returns True if this compartment has the same root name as another.
        """
        if type(comp) is str:
            return self.name == comp
        else:
            return self.name == comp.name

    def has_name_in_list(self, comps: list) -> bool:
        """
        Returns True if this compartment has the same root name
        as any compartment in the provided list.
        """
        return any(self.has_name(c) for c in comps)

    def stratify(self, stratify_name: str, stratum_name: str):
        """
        Returns a copy of the Compartment with a new stratification applied to it.

        Args:
            stratify_name: The name of the new stratification.
            stratum_name: The name of the strata that will be given to the new Compartment.
        Returns:
            Compartment: The new, stratified compartment.

        """
        new_strata = {**self.strata, stratify_name: stratum_name}
        return Compartment(
            name=self.name,
            strata=new_strata,
        )

    def serialize(self) -> str:
        """Returns a string representation of the compartment"""
        strata_strings = [f"{k}_{v}" for k, v in self.strata.items()]
        return "X".join([self.name, *strata_strings])

    @staticmethod
    def deserialize(s: str):
        """Converts a string into a Compartment"""
        parts = s.split("X")
        name = parts[0]
        strata = {}
        for strat in parts[1:]:
            stratum_parts = strat.split("_")
            s_name = stratum_parts[0]
            s_value = "_".join(stratum_parts[1:])
            strata[s_name] = s_value

        return Compartment(name, strata)

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
