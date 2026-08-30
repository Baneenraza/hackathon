"""Block 9 - FactorySimulator digital twin.

Three what-if scenarios over a planning horizon, all driven by the REAL
`failure_prob` (tabular classifier) and `rul_estimate` (LSTM) coming from
PredictiveAgent - no hardcoded outcome numbers.

Each scenario returns: expected_units, downtime_hours, cost, risk_score.
`compare()` returns a tidy DataFrame across the three.

Economic assumptions (documented, tunable via the constructor):
    units_per_hour            nominal throughput of the cell
    downtime_cost_per_hour    lost-margin + idle-labour while the cell is down
    unplanned_repair_hours    MTTR for a failure that happens in production
    planned_stop_hours        MTTR for a maintenance stop scheduled now
    catastrophic_cost         extra cost of an in-production failure (scrap,
                              collateral damage, expedited parts)
    planned_maintenance_cost  parts + labour for a scheduled intervention
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FactorySimulator:
    failure_prob: float                # 0..1 from the tabular classifier
    rul_estimate: float                # cycles/hours from the LSTM
    horizon_hours: float = 168.0       # 1 week planning horizon
    units_per_hour: float = 60.0
    downtime_cost_per_hour: float = 1500.0
    unplanned_repair_hours: float = 14.0
    planned_stop_hours: float = 4.0
    catastrophic_cost: float = 25000.0
    planned_maintenance_cost: float = 6000.0
    opex_per_hour: float = 400.0
    reduced_load_factor: float = 0.6
    _base: dict = field(default_factory=dict)

    def __post_init__(self):
        self.failure_prob = float(np.clip(self.failure_prob, 0.0, 1.0))
        self.rul_estimate = float(max(self.rul_estimate, 0.0))

    # ---- hazard model: blend the classifier probability with the RUL margin ----
    def _p_fail_continue(self):
        H = self.horizon_hours
        # fraction of the horizon that lies BEYOND the predicted failure point
        if self.rul_estimate >= H:
            p_rul = 0.03
        else:
            p_rul = 1.0 - self.rul_estimate / H
        # combine independent-ish contributions
        p = 1.0 - (1.0 - self.failure_prob) * (1.0 - p_rul)
        return float(np.clip(p, 0.0, 0.98))

    # ---- scenarios ----
    def simulate_continue(self):
        p = self._p_fail_continue()
        downtime = p * self.unplanned_repair_hours
        units = self.units_per_hour * max(self.horizon_hours - downtime, 0)
        cost = (downtime * self.downtime_cost_per_hour
                + p * self.catastrophic_cost
                + self.horizon_hours * self.opex_per_hour)
        return self._pack("continue", units, downtime, cost, p)

    def simulate_maintenance_stop(self):
        downtime = self.planned_stop_hours
        # a timely intervention resets the machine to a healthy baseline
        p = 0.02
        units = self.units_per_hour * max(self.horizon_hours - downtime, 0)
        cost = (downtime * self.downtime_cost_per_hour
                + self.planned_maintenance_cost
                + (self.horizon_hours - downtime) * self.opex_per_hour)
        return self._pack("maintenance_stop", units, downtime, cost, p)

    def simulate_reduced_load(self):
        # lower stress roughly halves the failure hazard, throughput scales down
        p = float(np.clip(self._p_fail_continue() * 0.5, 0.0, 0.9))
        downtime = p * self.unplanned_repair_hours
        rate = self.units_per_hour * self.reduced_load_factor
        units = rate * max(self.horizon_hours - downtime, 0)
        cost = (downtime * self.downtime_cost_per_hour
                + p * self.catastrophic_cost
                + self.horizon_hours * self.opex_per_hour * 0.9)
        return self._pack("reduced_load", units, downtime, cost, p)

    def _pack(self, name, units, downtime, cost, risk):
        return {"scenario": name,
                "expected_units": round(units, 1),
                "downtime_hours": round(downtime, 2),
                "cost": round(cost, 2),
                "risk_score": round(risk, 4)}

    def run_all(self):
        return [self.simulate_continue(),
                self.simulate_maintenance_stop(),
                self.simulate_reduced_load()]

    def compare(self):
        df = pd.DataFrame(self.run_all()).set_index("scenario")
        df["net_value"] = (df["expected_units"] * self._margin_per_unit()
                           - df["cost"]).round(2)
        return df

    def _margin_per_unit(self):
        return 40.0

    def recommend(self):
        """Pick the scenario with the best risk-adjusted net value."""
        df = self.compare()
        # penalise risk explicitly so a slightly richer but risky plan loses
        df["adjusted"] = df["net_value"] - df["risk_score"] * self.catastrophic_cost
        best = df["adjusted"].idxmax()
        return best, df


if __name__ == "__main__":
    sim = FactorySimulator(failure_prob=0.62, rul_estimate=90)
    best, df = sim.recommend()
    print(df.to_string())
    print("\nrecommended scenario:", best)
