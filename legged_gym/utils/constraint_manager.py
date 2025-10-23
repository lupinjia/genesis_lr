import torch

class ConstraintManager:
    """Handle the computation of termination probabilities based on constraints
    violations (Constraints as Terminations). refer to https://constraints-as-terminations.github.io/

    Args:
        tau (float): discount factor
        min_p (float): minimum termination probability
    """

    def __init__(self, tau=0.95, min_p=0.0):
        self.running_maxes = {}  # Polyak average of the maximum constraint violation
        self.running_mins = {}  # Polyak average of the minimum constraint violation
        self.probs = {}  # Termination probabilities for each constraint
        self.tau = tau  # Discount factor
        self.min_p = min_p  # Minimum termination probability

    def reset(self):
        """Reset the termination probabilities of the constraint manager."""
        self.probs = {}

    def add(self, name, constraint, max_p=0.1):
        """Add a constraint violation to the constraint manager and compute the
        associated termination probability.

        Args:
            name (string): name of the constraint
            constraint (float tensor): value of constraint violations for this constraint
            max_p (float): maximum termination probability
        """

        # First, put constraint in the form Torch.FloatTensor((num_envs, n_constraints))
        # Convert constraints violation to float if they are not
        if not torch.is_floating_point(constraint):
            constraint = constraint.float()

        # Ensure constraint is 2-dimensional even with a single element
        if len(constraint.size()) == 1:
            constraint = constraint.unsqueeze(1)

        # Get the maximum constraint violation for the current step
        constraint_max = constraint.max(dim=0, keepdim=True)[0].clamp(min=1e-6)

        # Compute polyak average of the maximum constraint violation for this constraint
        if name not in self.running_maxes:
            self.running_maxes[name] = constraint_max
        else:
            self.running_maxes[name] = (
                self.tau * self.running_maxes[name] + (1.0 - self.tau) * constraint_max
            )

        # Get samples for which there is a constraint violation
        mask = constraint > 0.0

        # Compute the termination probability which scales between min_p and max_p with
        # increasing constraint violation. Remains at 0 when there is no violation.
        probs = torch.zeros_like(constraint)
        probs[mask] = self.min_p + torch.clamp(
            constraint[mask]
            / (self.running_maxes[name].expand(constraint.size())[mask]),
            min=0.0,
            max=1.0,
        ) * (max_p - self.min_p)
        self.probs[name] = probs

    def get_probs(self):
        """Returns the termination probabilities due to constraint violations."""
        probs = torch.cat(list(self.probs.values()), dim=1)
        probs = probs.max(1).values
        return probs

    def get_str(self, names=None):
        """Get a debug string with constraints names and their average termination probabilities"""
        if names is None:
            names = list(self.probs.keys())
        txt = ""
        for name in names:
            txt += " {}: {}".format(
                name,
                str(
                    100.0 * self.probs[name].max(1).values.gt(0.0).float().mean().item()
                )[:4],
            )
            # txt += " {}: {}".format(name, str(100.0*self.probs[name].max(1).values.float().mean().item())[:4])

        return txt[1:]

    def log_all(self, episode_sums):
        """Log terminations probabilities in episode_sums with cstr_NAME key."""
        for name in list(self.probs.keys()):
            values = self.probs[name].max(1).values.gt(0.0).float()
            if "cstr_" + name not in episode_sums:
                episode_sums["cstr_" + name] = torch.zeros_like(values)
            episode_sums["cstr_" + name] += values

    def get_names(self):
        """Return a list of all constraint names."""
        return list(self.probs.keys())

    def get_vals(self):
        """Return a list of all constraint termination probabilities."""
        res = []
        for key in self.probs.keys():
            res += [100.0 * self.probs[key].max(1).values.gt(0.0).float().mean().item()]
        return res
