"""The proposed experiment as a table for whoever will run it.

A design that is right and is then programmed wrongly at the instrument is a
wasted measurement.  The table has to say exactly what to do: which delays, in
what units, how long a repetition takes, and how to share the time.  And it
has to state the timing convention, because a sequence built with tau as the
full pulse spacing measures at half the delays intended.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuclear_spin_recovery import (
    Experiment,
    SequenceDuration,
    format_measurement_table,
    measurement_rows,
    measurement_table_html,
    write_measurement_csv,
)
from nuclear_spin_recovery.design import TIMING_NOTE


@pytest.fixture
def design():
    """CPMG-16 at three delays, 0.5, 1 and 2 us."""
    return Experiment(tau=np.array([0.5e-3, 1e-3, 2e-3]), n_pulses=16,
                      b_z=311.0, sigma=0.01, weight=np.array([4.0, 2.0, 1.0]))


# -------------------------------------------------------------------- rows

def test_one_row_per_delay_in_nanoseconds(design):
    rows = measurement_rows(design)
    assert [r[0] for r in rows] == pytest.approx([500.0, 1000.0, 2000.0])


def test_a_repetition_lasts_two_n_tau_in_microseconds(design):
    rows = measurement_rows(design)
    assert [r[1] for r in rows] == pytest.approx([16.0, 32.0, 64.0])


def test_shares_sum_to_one(design):
    rows = measurement_rows(design)
    assert sum(r[2] for r in rows) == pytest.approx(1.0)
    assert sum(r[3] for r in rows) == pytest.approx(1.0)


def test_time_and_repetition_shares_differ_by_the_duration(design):
    """Weights 4, 2, 1 at durations 16, 32, 64 us: equal time at each delay,
    and repetitions in the ratio 4 : 2 : 1."""
    rows = measurement_rows(design)
    assert [r[2] for r in rows] == pytest.approx([1 / 3] * 3)
    assert [r[3] for r in rows] == pytest.approx([4 / 7, 2 / 7, 1 / 7])


def test_noise_is_the_design_noise_at_each_weight(design):
    rows = measurement_rows(design)
    assert [r[4] for r in rows] == pytest.approx([0.005, 0.01 / np.sqrt(2), 0.01])


def test_rows_come_out_in_increasing_delay():
    shuffled = Experiment(tau=np.array([2e-3, 0.5e-3, 1e-3]), n_pulses=16,
                          b_z=311.0, sigma=0.01, weight=np.array([1.0, 4.0, 2.0]))
    rows = measurement_rows(shuffled)
    assert [r[0] for r in rows] == pytest.approx([500.0, 1000.0, 2000.0])
    assert [r[3] for r in rows] == pytest.approx([4 / 7, 2 / 7, 1 / 7])


def test_an_unweighted_experiment_is_equal_repetitions():
    plain = Experiment(tau=np.array([1e-3, 2e-3]), n_pulses=8, b_z=311.0)
    rows = measurement_rows(plain)
    assert [r[3] for r in rows] == pytest.approx([0.5, 0.5])
    assert [r[2] for r in rows] == pytest.approx([1 / 3, 2 / 3])
    assert all(np.isnan(r[4]) for r in rows)


def test_the_cost_model_of_the_design_can_be_given(design):
    rows = measurement_rows(design, cost=SequenceDuration(overhead=10e-3))
    assert [r[1] for r in rows] == pytest.approx([26.0, 42.0, 74.0])


def test_an_experiment_that_measures_nothing_raises():
    empty = Experiment(tau=np.array([1e-3]), n_pulses=8, b_z=311.0,
                       weight=np.array([0.0]))
    with pytest.raises(ValueError, match="nothing"):
        measurement_rows(empty)


# -------------------------------------------------------------------- text

def test_the_text_table_says_what_to_run(design):
    text = format_measurement_table(design, gain=1.234)
    assert "CPMG-16" in text
    assert "311 G" in text
    assert "0.50 to 2.00 us" in text
    assert "1.23 nats" in text
    assert text.count("\n") >= 10


def test_every_output_states_the_timing_convention(design, tmp_path):
    assert "2 tau apart" in TIMING_NOTE
    assert TIMING_NOTE in format_measurement_table(design)
    assert TIMING_NOTE in measurement_table_html(design)


def test_the_budget_is_quoted_against_a_reference_time(design):
    """The design spends 3 x 0.064 ms; against 1.92 ms that is 10%."""
    text = format_measurement_table(design, reference_time=1.92)
    assert "10.00% of the reference experiment" in text
    assert "reference experiment" not in format_measurement_table(design)


def test_the_html_table_has_a_row_per_delay(design):
    html = measurement_table_html(design, gain=1.0)
    assert html.count("<tr>") == 3 + 5 + 1      # delays, facts, header
    assert "Next measurement" in html
    assert "delay tau (ns)" in html


# --------------------------------------------------------------------- csv

def test_csv_round_trips_the_rows(design, tmp_path):
    path = write_measurement_csv(design, tmp_path / "next.csv")
    lines = path.read_text().splitlines()
    assert lines[0] == ("n_pulses,b_z_gauss,delay_tau_ns,repetition_us,"
                        "share_of_time,share_of_repetitions,expected_noise")
    assert len(lines) == 4
    table = np.array([[float(x) for x in line.split(",")] for line in lines[1:]])
    assert np.all(table[:, 0] == 16) and np.all(table[:, 1] == 311)
    np.testing.assert_allclose(table[:, 2:],
                               np.array(measurement_rows(design)), atol=1e-4)


def test_every_csv_line_names_its_sequence(design, tmp_path):
    """A line cut from the file still says which sequence it belongs to."""
    path = write_measurement_csv(design, tmp_path / "next.csv")
    assert all(line.startswith("16,311,")
               for line in path.read_text().splitlines()[1:])
