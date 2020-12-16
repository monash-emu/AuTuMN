from summer2.compartment import Compartment


def test_setup__with_no_strat():
    c = Compartment("infected")
    assert c == "infected"
    assert str(c) == "infected"
    assert hash(c) == hash("infected")


def test_setup__with_strat():
    c = Compartment("infected", {"location": "hawaii"})
    assert c == "infectedXlocation_hawaii"
    assert str(c) == "infectedXlocation_hawaii"
    assert hash(c) == hash("infectedXlocation_hawaii")


def test_setup__with_double_strat():
    c = Compartment("infected", {"location": "hawaii", "age": "15"})
    assert c == "infectedXlocation_hawaiiXage_15"
    assert str(c) == "infectedXlocation_hawaiiXage_15"
    assert hash(c) == hash("infectedXlocation_hawaiiXage_15")


def test_has_stratum():
    c = Compartment("infected", {"location": "hawaii", "age": "15"})
    assert c.has_stratum("location", "hawaii")
    assert c.has_stratum("age", "15")
    assert not c.has_stratum("location", "london")
    assert not c.has_stratum("age", "20")
    assert not c.has_stratum("clinical", "hospital")


def test_is_match():
    c = Compartment("infected", {"location": "hawaii", "age": "15"})
    assert c.is_match("infected", {})
    assert not c.is_match("susceptible", {})
    assert c.is_match("infected", {"age": "15"})
    assert c.is_match("infected", {"location": "hawaii"})
    assert c.is_match("infected", {"age": "15", "location": "hawaii"})
    assert not c.is_match("infected", {"age": "15", "location": "hawaii", "clinical": "hospital"})
    assert not c.is_match("infected", {"clinical": "hospital"})


def test_serialize():
    c = Compartment("infected", {"location": "hawaii", "age": "15"})
    assert c.serialize() == "infectedXlocation_hawaiiXage_15"


def test_deserialzie():
    s = "infectedXlocation_hawaiiXage_15"
    c = Compartment.deserialize(s)
    assert c == Compartment("infected", {"location": "hawaii", "age": "15"})
    assert c != Compartment("infected", {"age": "15", "location": "hawaii"})
    assert str(c) == "infectedXlocation_hawaiiXage_15"
    assert hash(c) == hash("infectedXlocation_hawaiiXage_15")


def test_has_name_in_list():
    c = Compartment("infected")
    assert c.has_name_in_list(["infected", "foo", "bar"])
    assert c.has_name_in_list([Compartment("infected"), Compartment("foo"), Compartment("bar")])
    assert not c.has_name_in_list(["infected_stuff", "foo", "bar"])
    assert not c.has_name_in_list(
        [Compartment("infected_stuff"), Compartment("foo"), Compartment("bar")]
    )


def test_has_name_in_list_stratified():
    c = Compartment("infected", {"location": "hawaii", "age": "15"})
    assert c.has_name_in_list(["infected", "foo", "bar"])
    assert c.has_name_in_list([Compartment("infected"), Compartment("foo"), Compartment("bar")])
    assert not c.has_name_in_list(["infected_stuff", "foo", "bar"])
    assert not c.has_name_in_list(
        [Compartment("infected_stuff"), Compartment("foo"), Compartment("bar")]
    )


def test_stratify():
    c = Compartment("infected")
    assert c.name == "infected"
    assert c.strata == {}
    assert c.serialize() == "infected"
    c_age = c.stratify("age", "15")
    assert c_age.name == "infected"
    assert c_age.strata == {"age": "15"}
    assert c_age.serialize() == "infectedXage_15"
    c_loc = c_age.stratify("location", "work")
    assert c_loc.name == "infected"
    assert c_loc.strata == {"age": "15", "location": "work"}
    assert c_loc.serialize() == "infectedXage_15Xlocation_work"
