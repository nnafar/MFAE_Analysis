# -*- coding: utf-8 -*-
"""
Experiment folder naming rule: the single definition used by the
per-experiment analysis (to read pulse settings) and the bulk analysis (to
group experiments into conditions).

    YYMMDD_CellType_Treatment_Chip_Experiment-<P>Pa-<V>V-<D><ms|us|s>-frame<N>
    260128_MDAMB231_WT_Chip1_Experiment3-1100Pa-100V-1ms-frame10     electroporated
    260326_MDAMB231_WT_Chip2_Experiment283-1100Pa-0V-0ms-frame0      aspiration only
"""

import re
from dataclasses import dataclass
from typing import Optional

FOLDER_PATTERN = re.compile(
    r"(\d{6})_([^_\-]+)_([^_\-]+)_([^_\-]+)_([^_\-]+)"
    r"[-_](\d+)Pa"
    r"[-_](\d+)V"
    r"[-_](\d+)(ms|us|µs|s)"
    r"[-_]frame(\d+)",
    re.IGNORECASE,
)

NAMING_RULE = "YYMMDD_Cell_Treatment_Chip_Experiment-<P>Pa-<V>V-<D><ms|us|s>-frame<N>"


@dataclass(frozen=True)
class ExperimentName:
    date: str
    cell_type: str
    treatment: str
    chip: str
    experiment: str
    pressure_pa: int
    voltage_v: int
    duration_value: float
    duration_unit: str          # 'ms', 'us' or 's'
    pulse_frame: int

    @property
    def has_pulse(self) -> bool:
        """0 V means aspiration only."""
        return self.voltage_v > 0

    @property
    def duration_ms(self) -> float:
        factor = {'s': 1000.0, 'ms': 1.0, 'us': 1e-3}[self.duration_unit]
        return self.duration_value * factor

    @property
    def duration_label(self) -> str:
        value = int(self.duration_value) if float(self.duration_value).is_integer() else self.duration_value
        return f"{value}{self.duration_unit}"

    def describe(self) -> str:
        if not self.has_pulse:
            return f"{self.pressure_pa} Pa, no pulse (aspiration only)"
        return (f"{self.pressure_pa} Pa, {self.voltage_v} V, {self.duration_label} pulse "
                f"at frame {self.pulse_frame}")


def parse_experiment_name(name: str) -> Optional[ExperimentName]:
    """Parse a folder name / experiment_id; None if it doesn't follow the naming rule."""
    match = FOLDER_PATTERN.match(str(name))
    if not match:
        return None
    unit = match.group(9).lower()
    if unit in ('µs', 'μs'):
        unit = 'us'
    return ExperimentName(
        date=match.group(1),
        cell_type=match.group(2),
        treatment=match.group(3),
        chip=match.group(4),
        experiment=match.group(5),
        pressure_pa=int(match.group(6)),
        voltage_v=int(match.group(7)),
        duration_value=float(match.group(8)),
        duration_unit=unit,
        pulse_frame=int(match.group(10)),
    )
