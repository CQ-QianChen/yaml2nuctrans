import numpy as np
import radioactivedecay as rd

from yaml2nuctrans.post_processing.repository_discharge import (
    DOMAIN_ZONE,
    HOST_ROCK_ZONE,
    SECONDS_PER_YEAR,
    compute_ewg_discharge,
)

# Nuclides chosen only for convenient half-lives relative to the test time spans below.
NUCLIDE_A = "I-129"  # half-life ~15.7 million years
NUCLIDE_B = "Cl-36"  # half-life ~301,000 years


def _decay_constant_per_year(nuclide_name):
    return np.log(2) / rd.Nuclide(nuclide_name).half_life() * SECONDS_PER_YEAR


def _substance_amounts(nuclide_name, initial_amount, times_years, host_rock_amount):
    """Build the `compute_zone_substance_amounts` shaped input for one
    nuclide"""
    times_s = np.asarray(times_years) * SECONDS_PER_YEAR
    zone_series = {}
    for t, n in zip(times_s, host_rock_amount):
        zone_series[(DOMAIN_ZONE, t)] = [initial_amount]
        zone_series[(HOST_ROCK_ZONE, t)] = [n]
    return {f"{nuclide_name}_substance_amount": zone_series}


def _field(results, nuclide_name, field_name, times_years):
    """return results a flat array"""
    times_s = np.asarray(times_years) * SECONDS_PER_YEAR
    series = results[f"{nuclide_name}_{field_name}"]
    return np.array([series[(HOST_ROCK_ZONE, t)][0] for t in times_s])


def test_zero_discharge_under_pure_decay_no_transport():
    """If no transport, the mass-balance identity `discharged = N(0) - N(t) - cumulative_decay_loss`
    should recover ~0 at every time
    """
    initial_amount = 1000.0
    times_years = np.linspace(0.0, 5_000_000.0, 51)
    decay_constant = _decay_constant_per_year(NUCLIDE_A)
    host_rock_amount = initial_amount * np.exp(-decay_constant * times_years)

    substance_amounts = _substance_amounts(
        NUCLIDE_A, initial_amount, times_years, host_rock_amount
    )
    results = compute_ewg_discharge(
        substance_amounts,
        domain_zone=DOMAIN_ZONE,
        host_rock_zone=HOST_ROCK_ZONE,
        fields=("discharged", "frac_discharged", "release_rate"),
    )

    discharged = _field(results, NUCLIDE_A, "discharged", times_years)
    frac_discharged = _field(results, NUCLIDE_A, "frac_discharged", times_years)
    release_rate = _field(results, NUCLIDE_A, "release_rate", times_years)

    # `discharged` is clipped to >= 0 by construction, so under pure decay
    # it should land exactly at the floor.
    np.testing.assert_allclose(discharged, 0.0, atol=1e-2)
    np.testing.assert_allclose(frac_discharged, 0.0, atol=1e-5)
    np.testing.assert_allclose(release_rate, 0.0, atol=1e-8)


def test_discharge_matches_mass_lost_when_decay_is_negligible():
    """when the time span is much smaller than the nuclide's half-life, decay removes a negligible fraction of the
    inventory, so essentially all of a prescribed host-rock decline must
    be attributed to transport: `discharged(t) ~= N(0) - N(t)`.
    """
    initial_amount = 1000.0
    end_time_years = 10_000.0  # much smaller than I-129's ~15.7 million year half-life
    times_years = np.linspace(0.0, end_time_years, 51)
    # synthetic transport-driven decline (not tied to any decay law).
    host_rock_amount = initial_amount - 0.5 * initial_amount * (
        times_years / end_time_years
    )

    substance_amounts = _substance_amounts(
        NUCLIDE_A, initial_amount, times_years, host_rock_amount
    )
    results = compute_ewg_discharge(
        substance_amounts,
        domain_zone=DOMAIN_ZONE,
        host_rock_zone=HOST_ROCK_ZONE,
        fields=("discharged",),
    )

    discharged = _field(results, NUCLIDE_A, "discharged", times_years)
    naive_mass_lost = host_rock_amount[0] - host_rock_amount

    np.testing.assert_allclose(discharged, naive_mass_lost, atol=1.0)


def test_total_aggregates_sum_across_nuclides():
    """The `total_*` entries should sum the per-nuclide amounts."""
    initial_amount_a = 1000.0
    initial_amount_b = 400.0
    times_years = np.linspace(0.0, 10_000.0, 11)
    host_rock_amount_a = initial_amount_a * np.linspace(1.0, 0.5, 11)
    host_rock_amount_b = initial_amount_b * np.linspace(1.0, 0.2, 11)

    substance_amounts = {
        **_substance_amounts(
            NUCLIDE_A, initial_amount_a, times_years, host_rock_amount_a
        ),
        **_substance_amounts(
            NUCLIDE_B, initial_amount_b, times_years, host_rock_amount_b
        ),
    }
    results = compute_ewg_discharge(
        substance_amounts,
        domain_zone=DOMAIN_ZONE,
        host_rock_zone=HOST_ROCK_ZONE,
        fields=(
            "discharged",
            "frac_discharged",
            "total_discharged",
            "total_frac_discharged",
        ),
    )

    discharged_a = _field(results, NUCLIDE_A, "discharged", times_years)
    discharged_b = _field(results, NUCLIDE_B, "discharged", times_years)
    total_discharged = _field(results, "total", "discharged", times_years)
    total_frac_discharged = _field(results, "total", "frac_discharged", times_years)

    np.testing.assert_allclose(total_discharged, discharged_a + discharged_b)
    np.testing.assert_allclose(
        total_frac_discharged,
        total_discharged / (initial_amount_a + initial_amount_b),
    )
