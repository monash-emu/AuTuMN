
def make_voc_seed_func(entry_rate, start_time, seed_duration):
    """
    Create a simple step function to allow seeding of the VoC strain at a particular point in time.
    """

    def voc_seed_func(time, computed_values):
        return entry_rate if 0. < time - start_time < seed_duration else 0.

    return voc_seed_func
