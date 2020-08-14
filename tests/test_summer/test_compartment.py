from summer.compartment import Compartment


def test_setup__with_no_strat():
    c = Compartment("infected")
    assert str(c) == "infected"
    assert hash(c) == hash("infected")


def test_setup__with_strat():
    c = Compartment("infected", strat_names=["location"], strat_values={"location": "hawaii"})
    assert str(c) == "infectedXlocation_hawaii"
    assert hash(c) == hash("infectedXlocation_hawaii")


def test_setup__with_double_strat():
    c = Compartment(
        "infected",
        strat_names=["location", "age"],
        strat_values={"location": "hawaii", "age": "15"},
    )
    assert str(c) == "infectedXlocation_hawaiiXage_15"
    assert hash(c) == hash("infectedXlocation_hawaiiXage_15")


def test_serialize():
    c = Compartment(
        "infected",
        strat_names=["location", "age"],
        strat_values={"location": "hawaii", "age": "15"},
    )
    assert c.serialize() == "infectedXlocation_hawaiiXage_15"


def test_deserialzie():
    s = "infectedXlocation_hawaiiXage_15"
    c = Compartment.deserialize(s)
    assert c == Compartment(
        "infected",
        strat_names=["location", "age"],
        strat_values={"location": "hawaii", "age": "15"},
    )
    assert c != Compartment(
        "infected",
        strat_names=["age", "location"],
        strat_values={"location": "hawaii", "age": "15"},
    )
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
    c = Compartment(
        "infected",
        strat_names=["location", "age"],
        strat_values={"location": "hawaii", "age": "15"},
    )
    assert c.has_name_in_list(["infected", "foo", "bar"])
    assert c.has_name_in_list([Compartment("infected"), Compartment("foo"), Compartment("bar")])
    assert not c.has_name_in_list(["infected_stuff", "foo", "bar"])
    assert not c.has_name_in_list(
        [Compartment("infected_stuff"), Compartment("foo"), Compartment("bar")]
    )


def test_stratify():
    c = Compartment("infected")
    assert c._name == "infected"
    assert c._strat_names == tuple()
    assert c._strat_values == {}
    assert c.serialize() == "infected"
    c_age = c.stratify("age", "15")
    assert c_age._name == "infected"
    assert c_age._strat_names == ("age",)
    assert c_age._strat_values == {"age": "15"}
    assert c_age.serialize() == "infectedXage_15"
    c_loc = c_age.stratify("location", "work")
    assert c_loc._name == "infected"
    assert c_loc._strat_names == ("age", "location")
    assert c_loc._strat_values == {"age": "15", "location": "work"}
    assert c_loc.serialize() == "infectedXage_15Xlocation_work"
