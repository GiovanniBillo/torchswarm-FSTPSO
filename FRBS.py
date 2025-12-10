import time
import copy
from enum import Enum

import torch
import numpy as np

from torchswarm.utils.parameters import SwarmParameters
from debug_utils import _vprint


class DELTA(Enum):
    SAME = 0
    FAR = 1
    NEAR = 2


class PHI(Enum):
    BETTER = 0
    SAME = 1
    WORSE = 2


class OUT(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class Frbs:
    """
    Fuzzy rule-based system (FRBS) for updating PSO parameters.
    """

    def __init__(self, delta_max, verbose=False):
        self.delta_max = float(delta_max)
        self.d1 = 0.2 * self.delta_max
        self.d2 = 0.4 * self.delta_max
        self.d3 = 0.6 * self.delta_max

        self.verbose = verbose

        # Defuzzification values z (as tensor)
        # z_tensor = torch.tensor(
        #     [
        #         [0.3, 0.5, 1.0],    # Inertia
        #         [1.0, 2.0, 3.0],    # Social
        #         [0.1, 1.5, 3.0],    # Cognitive
        #         [0.0, 0.001, 0.01], # L
        #         [0.1, 0.15, 0.2],   # U
        #     ],
        #     dtype=torch.float32,
        # )
        # z_tensor = torch.tensor(
        #             [
        #                 [0.3, 0.3, 0.5, 0.5, 1.0, 1.0],    # Inertia
        #                 [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],    # Social
        #                 [0.1, 1.5, 1.5, 1.5, 1.5, 3.0],    # Cognitive
        #                 [0.0, 0.0, 0.0, 0.001, 0.001, 0.01], # L
        #                 [0.1, 0.15, 0.15,, 0.15, 0.15, 0.2],   # U
        #             ],
        #             dtype=torch.float32,
        #         )
        z_list = [ 
            torch.Tensor([0.3, 0.3, 0.5, 0.5, 1.0, 1.0]),    # Inertia
            torch.Tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),    # Social
            torch.Tensor([0.1, 1.5, 1.5, 1.5, 1.5, 3.0]),    # Cognitive
            torch.Tensor([0.0, 0.0, 0.0, 0.001, 0.001, 0.01]), # L
            torch.Tensor([0.1, 0.15, 0.15, 0.15, 0.15, 0.2]),   # U
                    ]
        self.components = ["Inertia", "Social", "Cognitive", "L", "U"]

        # One 1D tensor per component
        self.z_dict = {name: z_list[i] for i, name in enumerate(self.components)}

        self.rules = None
        self.phi_membership = None
        self.delta_membership = None

        print("Initialized Fuzzy Rule Based System")

    # ------------------------
    #   DELTA MEMBERSHIP
    # ------------------------
    def get_delta_membership(self, delta):
        """
        Compute fuzzy memberships for delta (SAME, NEAR, FAR).
        Returns tensors (scalar 0-D) for each membership.
        """
        delta = float(delta)
        assert delta <= self.delta_max, (
            f"AssertionError: delta={delta} is larger than "
            f"delta_max={self.delta_max}"
        )

        def Same():
            if 0.0 <= delta < self.d1:
                return torch.tensor(1.0, dtype=torch.float32)
            elif self.d1 <= delta < self.d2:
                val = (self.d2 - delta) / (self.d2 - self.d1)
                return torch.tensor(val, dtype=torch.float32)
            elif self.d2 <= delta <= self.delta_max:
                return torch.tensor(0.0, dtype=torch.float32)
            else:
                _vprint(self.verbose, f"delta={delta} is outside all membership intervals [Same]")
                raise ValueError(f"delta={delta} is outside all membership intervals [Same]")

        def Near():
            if 0.0 <= delta < self.d1:
                return torch.tensor(0.0, dtype=torch.float32)
            elif self.d1 <= delta < self.d2:
                val = (delta - self.d1) / (self.d2 - self.d1)
                return torch.tensor(val, dtype=torch.float32)
            elif self.d2 <= delta < self.d3:
                val = (self.d3 - delta) / (self.d3 - self.d2)
                return torch.tensor(val, dtype=torch.float32)
            elif self.d3 <= delta <= self.delta_max:
                return torch.tensor(0.0, dtype=torch.float32)
            else:
                _vprint(self.verbose, f"delta={delta} is outside all membership intervals [Near]")
                raise ValueError(f"delta={delta} is outside all membership intervals [Near]")

        def Far():
            if 0.0 <= delta < self.d2:
                return torch.tensor(0.0, dtype=torch.float32)
            elif self.d2 <= delta < self.d3:
                val = (delta - self.d2) / (self.d3 - self.d2)
                return torch.tensor(val, dtype=torch.float32)
            elif self.d3 <= delta <= self.delta_max:
                return torch.tensor(1.0, dtype=torch.float32)
            else:
                _vprint(self.verbose, f"delta={delta} is outside all membership intervals [Far]")
                raise ValueError(f"delta={delta} is outside all membership intervals [Far]")

        self.delta_membership = {
            "Same": Same(),
            "Near": Near(),
            "Far":  Far(),
        }

        return self.delta_membership

    # ------------------------
    #   PHI MEMBERSHIP
    # ------------------------
    def get_phi_membership(self, phi):
        """
        Compute fuzzy memberships for phi (BETTER, SAME, WORSE).
        phi is expected in [-1, 1].
        """
        phi = float(phi)

        def Same():
            # Usually a triangular-like membership centered at 0
            val = 1.0 - abs(phi)
            val = max(0.0, min(1.0, val))
            return torch.tensor(val, dtype=torch.float32)

        def Worse():
            # e.g. positive phi → worse
            if -1.0 <= phi < 0.0:
                return torch.tensor(0.0, dtype=torch.float32)
            elif 0.0 <= phi < 1.0:
                return torch.tensor(phi, dtype=torch.float32)
            elif phi == 1.0:
                return torch.tensor(1.0, dtype=torch.float32)
            else:
                raise ValueError(f"phi={phi} outside expected range [-1,1] for Worse")

        def Better():
            # e.g. negative phi → better
            if phi == -1.0:
                return torch.tensor(1.0, dtype=torch.float32)
            elif -1.0 < phi < 0.0:
                return torch.tensor(-phi, dtype=torch.float32)
            elif 0.0 <= phi <= 1.0:
                return torch.tensor(0.0, dtype=torch.float32)
            else:
                raise ValueError(f"phi={phi} outside expected range [-1,1] for Better")

        self.phi_membership = {
            "Same":   Same(),
            "Worse":  Worse(),
            "Better": Better(),
        }

        return self.phi_membership

    # ------------------------
    #   MEMBERSHIPS WRAPPER
    # ------------------------
    def compute_memberships(self, delta, phi):
        delta_m = self.get_delta_membership(delta)
        phi_m = self.get_phi_membership(phi)
        return delta_m, phi_m

    # ------------------------
    #   RULE DEFINITION
    # ------------------------
    def define_rules(self, delta_membership, phi_membership):
        """
        Build the fuzzy rule base using delta_membership and phi_membership.
        All rule activations are scalar tensors.
        """
        if delta_membership is None or phi_membership is None:
            raise RuntimeError("Memberships not computed. Call compute_memberships(delta, phi) first.")

        delta = delta_membership
        phi = phi_membership

        self.rules = {
            "Inertia": torch.tensor([
                phi["Worse"], delta["Same"],   # Rule 1: Low
                phi["Same"],  delta["Near"],   # Rule 2: Medium
                phi["Better"], delta["Far"],   # Rule 3: High
            ]),

            "Social": torch.tensor([
                phi["Better"], delta["Near"],  # Rule 4: Low
                phi["Same"],   delta["Same"],  # Rule 5: Medium
                phi["Worse"],  delta["Far"],   # Rule 6: High
            ]),

            "Cognitive": torch.tensor([
                delta["Far"],                                  # Rule 7: Low
                phi["Worse"],                                  # RUle 8: Medium
                phi["Same"],
                delta["Same"],
                delta["Near"],    
                phi["Better"],                                 # Rule 9: High
            ]),

            "L": torch.tensor([
                phi["Same"],                                    # Rule 10: Low
                phi["Better"],
                delta["Far"],  
                delta["Same"],                                  # Rule 11: Medium
                delta["Near"],           
                phi["Worse"],                                   # Rule 12: High
            ]),

            "U": torch.tensor([
                delta["Same"],                                   # Rule 13: Low
                phi["Same"],                                     # Rule 14: Medium 
                phi["Better"], 
                delta["Near"], 
                phi["Worse"], delta["Far"],                      # Rule 15: High
            ]),
        }
        return self.rules
    # ------------------------
    #   SUGENO DEFUZZIFICATION
    # ------------------------
    def sugeno(self, rules):
        """
        Sugeno-type defuzzification:
        OUT[c] = sum_i (rule_i * z_i) / sum_i (rule_i)
        """
        if self.rules is None:
            raise RuntimeError("Rules not defined. Call define_rules() first.")

        OUT = {name: torch.tensor(0.0, dtype=torch.float32) for name in self.components}

        for c in self.components:
            for i, r in enumerate(self.rules[c]):
                # r: rule activation (tensor scalar)
                # self.z_dict[c][i]: scalar weight
                OUT[c] += r * self.z_dict[c][i]

            denum = torch.sum(self.rules[c], dim=0)
            if denum.item() == 0:
                OUT[c] = torch.tensor(0.0)
            else:
                OUT[c] /= denum

        return OUT

    # ------------------------
    #   (OPTIONAL) TEST / DEBUG
    # ------------------------
    def get_stuff(self):
        """
        Placeholder – not used yet.
        """
        raise NotImplementedError("get_stuff() is not implemented yet.")


def main():
    # simple test for the Fuzzy Rule Based System and Sugeno Method
    test_delta = 25.0      # any real number in [0, delta_max]
    test_phi = 0.3         # [-1, +1]
    delta_max = 50.0

    test_FRBS = Frbs(delta_max)
    memberships = test_FRBS.compute_memberships(test_delta, test_phi)
    print("memberships:", memberships)
    print("phi Worse:", memberships[1]["Worse"])

    test_FRBS.define_rules()
    print("rules:", test_FRBS.rules)

    out = test_FRBS.sugeno()
    print("Output of Sugeno method:", out)
    return 0


if __name__ == "__main__":
    main()

