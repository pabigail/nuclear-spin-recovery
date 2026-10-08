"""The proposed experiment, written for whoever will run it.

A design is an :class:`~nuclear_spin_recovery.experiment.Experiment` with
weights, which is what the sampler wants and not what a lab does.  The
functions here turn it into rows: one per delay, with the duration of a
repetition there and how the measurement time and the repetitions are to be
shared out.

**The timing convention is stated in every output**, because it is the thing
most easily misread at the instrument.  tau is the delay before the first pi
pulse and after the last; consecutive pi pulses are 2 tau apart.  That is the
convention of the forward model (see :mod:`~nuclear_spin_recovery.design.
cost`), and a sequence programmed with tau as the full spacing measures at
half the delays intended.

**Shares, not counts.**  The weights of a design are relative, so the table
gives the share of the measurement time and the share of the repetitions at
each delay.  The two differ because a long sequence takes longer per
repetition.  How many repetitions a share amounts to depends on the total time
available and the shot rate, which the design does not know.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .cost import SequenceDuration

#: The convention, in one sentence, as printed under every table.
TIMING_NOTE = (
    "tau is the delay before the first pi pulse and after the last; "
    "consecutive pi pulses are 2 tau apart.")

_COLUMNS = ("delay tau (ns)", "repetition (us)", "share of time",
            "share of repetitions", "expected noise")


def measurement_rows(experiment, cost=None):
    """One row per delay of ``experiment``, in increasing delay.

    Each row is ``(delay in ns, duration of one repetition in us, share of the
    measurement time, share of the repetitions, expected noise)``.  The
    duration is ``cost`` at that delay -- by default
    :class:`~nuclear_spin_recovery.design.cost.SequenceDuration` with no
    overhead, ``2 N tau``; pass the cost model the design was made with if it
    was another.  The noise is the standard deviation of the coherence the
    design assumed, ``sigma / sqrt(weight)``, and is ``nan`` if the experiment
    carries no sigma.

    An experiment without weights is read as equal weight at every delay.
    """
    cost = SequenceDuration() if cost is None else cost
    tau = np.asarray(experiment.tau, dtype=float)
    weight = (np.ones(tau.size) if experiment.weight is None
              else np.asarray(experiment.weight, dtype=float))
    if tau.size == 0 or not np.any(weight > 0):
        raise ValueError("the experiment measures nothing")
    duration = np.asarray(cost(experiment), dtype=float)
    time = weight * duration
    with np.errstate(divide="ignore"):
        noise = (np.full(tau.size, np.nan) if experiment.sigma is None
                 else experiment.sigma / np.sqrt(weight))
    order = np.argsort(tau, kind="stable")
    return [(float(tau[j] * 1e6), float(duration[j] * 1e3),
             float(time[j] / time.sum()), float(weight[j] / weight.sum()),
             float(noise[j])) for j in order]


def _facts(experiment, gain, reference_time, cost):
    cost = SequenceDuration() if cost is None else cost
    tau = np.asarray(experiment.tau, dtype=float)
    facts = [
        ("Sequence", f"CPMG-{experiment.n_pulses}"),
        ("Magnetic field", f"{experiment.b_z:g} G"),
        ("Delays to measure", f"{tau.size}"),
        ("Delay range", f"{tau.min() * 1e3:.2f} to {tau.max() * 1e3:.2f} us"),
    ]
    if reference_time is not None and experiment.weight is not None:
        total = float(np.sum(experiment.weight * cost(experiment)))
        share = f"{total / float(reference_time):.2%} of the reference experiment"
        facts.append(("Time budget designed for", share))
    if gain is not None:
        facts.append(("Expected information gain", f"{float(gain):.2f} nats"))
    return facts


def format_measurement_table(experiment, *, gain=None, reference_time=None,
                             cost=None):
    """The proposed experiment as plain text.

    ``gain`` is the expected information gain to quote, in nats.
    ``reference_time`` is a measurement time to express the design's budget
    against -- usually the time the first experiment took.
    """
    rows = measurement_rows(experiment, cost)
    facts = _facts(experiment, gain, reference_time, cost)
    width = max(len(name) for name, _ in facts)
    lines = ["Next measurement", ""]
    lines += [f"  {name:<{width}}  {value}" for name, value in facts]
    lines += ["", f"  {'#':>3s}  " + "  ".join(f"{c:>20s}" for c in _COLUMNS)]
    for i, (delay, duration, time, reps, noise) in enumerate(rows, start=1):
        lines.append(f"  {i:3d}  {delay:20.0f}  {duration:20.2f}  {time:20.1%}  "
                     f"{reps:20.1%}  {noise:20.3f}")
    lines += ["", f"  {TIMING_NOTE}"]
    return "\n".join(lines)


def measurement_table_html(experiment, *, gain=None, reference_time=None,
                           cost=None):
    """The same table as an HTML string, for a notebook.

    Wrap it in ``IPython.display.HTML`` to show it.  Returned as a string so
    that this module needs no notebook stack.
    """
    rows = measurement_rows(experiment, cost)
    facts = _facts(experiment, gain, reference_time, cost)
    cell = "padding:3px 14px;text-align:right;font-variant-numeric:tabular-nums"
    head = "padding:4px 14px;text-align:right;border-bottom:1.5px solid #888"
    summary = "".join(
        f"<tr><td style='padding:2px 14px 2px 0;color:#666'>{name}</td>"
        f"<td style='padding:2px 0'><b>{value}</b></td></tr>"
        for name, value in facts)
    header = "".join(f"<th style='{head}'>{c}</th>" for c in ("#", *_COLUMNS))
    body = "".join(
        f"<tr><td style='{cell}'>{i}</td><td style='{cell}'>{delay:.0f}</td>"
        f"<td style='{cell}'>{duration:.2f}</td><td style='{cell}'>{time:.1%}</td>"
        f"<td style='{cell}'>{reps:.1%}</td><td style='{cell}'>{noise:.3f}</td></tr>"
        for i, (delay, duration, time, reps, noise) in enumerate(rows, start=1))
    return (
        "<div style='font-family:sans-serif;font-size:14px'>"
        "<div style='font-size:17px;margin-bottom:6px'><b>Next measurement</b></div>"
        f"<table style='border-collapse:collapse;margin-bottom:10px'>{summary}</table>"
        f"<table style='border-collapse:collapse'><thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
        f"<div style='color:#666;margin-top:8px;max-width:640px'>{TIMING_NOTE} "
        "Divide the available measurement time between the delays by "
        "<i>share of time</i>; because a long sequence takes longer per "
        "repetition, that gives the repetition counts in <i>share of "
        "repetitions</i>. <i>Expected noise</i> is the standard deviation of "
        "the coherence at each delay for the budget above, and falls as the "
        "square root of any extra time.</div></div>")


def write_measurement_csv(experiment, path, cost=None):
    """Write the rows to ``path`` as CSV, one line per delay.  Returns the path.

    The pulse number and field are repeated on every line, so that a line cut
    from the file still says what sequence it belongs to.
    """
    header = ("n_pulses,b_z_gauss,delay_tau_ns,repetition_us,share_of_time,"
              "share_of_repetitions,expected_noise")
    lines = [header]
    lines += [f"{experiment.n_pulses},{experiment.b_z:g},{delay:.1f},"
              f"{duration:.4f},{time:.5f},{reps:.5f},{noise:.5f}"
              for delay, duration, time, reps, noise
              in measurement_rows(experiment, cost)]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path
