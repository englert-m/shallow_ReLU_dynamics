import argparse
import atexit
import copy
from datetime import datetime
import json
import math
import os
import numpy as np
import pandas as pd
import torch
import wandb
import hashlib
import sys

torch.set_default_dtype(torch.float64)


class Net(torch.nn.Module):
    def __init__(self, d, width, X, state_dict=None):
        super(Net, self).__init__()

        self.hidden = torch.nn.Linear(d, width, bias=False).to(device)
        torch.nn.init.normal_(self.hidden.weight, 0, 1 / math.sqrt(width * d))

        self.predict = torch.nn.Linear(width, 1, bias=False).to(device)

        # balanced initialization of output layer
        with torch.no_grad():
            neuron_norms = self.hidden.weight.norm(dim=1)
            output_layer_signs = (
                2 * torch.bernoulli(torch.ones_like(self.predict.weight) / 2) - 1
            )
            self.predict.weight.copy_(output_layer_signs * neuron_norms)

        # save the initial state of the network
        # but use supplied state_dict instead if one is provided
        self._initial_state = copy.deepcopy(
            self.state_dict() if state_dict is None else state_dict
        )

        # make sure that the weights from the supplied state_dict (if given) are used
        self.reset(1.0)

        # categorize neurons into three groups
        # which is useful for computing statistics later on
        with torch.no_grad():
            # find which neurons have a negative inner product with all inputs
            self.always_dead = torch.amax(self.hidden(X), dim=0) <= 0

            output_layer_signs = torch.sign(self.predict.weight.squeeze())

            self.negative_group = torch.logical_and(
                output_layer_signs < 0, ~self.always_dead
            )
            self.positive_group = torch.logical_and(
                output_layer_signs > 0, ~self.always_dead
            )

    def reset(self, scale=1.0):
        """Reset the network weights to a scaled version of the initial state."""
        self.load_state_dict(self._initial_state)
        with torch.no_grad():
            self.hidden.weight.copy_(self.hidden.weight * scale)
            self.predict.weight.copy_(self.predict.weight * scale)

    def forward(self, z):
        return self.predict(torch.relu(self.hidden(z)))

    def get_initialization_hash(self):
        """Return a hash of the initial weights."""
        return hashlib.blake2b(
            self._initial_state["hidden.weight"].cpu().numpy().tobytes(), digest_size=8
        ).hexdigest()

    def get_neuron_lengths(self):
        """ "Return the norms of the hidden layer neurons."""
        return torch.linalg.vector_norm(self.hidden.weight, dim=1)

    def get_neuron_directions(self):
        """Return the normalized hidden layer neurons."""
        with torch.no_grad():
            return torch.nn.functional.normalize(self.hidden.weight, dim=1)

    def get_neuron_lengths_change(self):
        """Return the gradients of the length of the hidden layer neurons."""
        with torch.no_grad():
            return -torch.sum(
                self.hidden.weight.grad * self.get_neuron_directions(), dim=1
            )  # TODO check and test

    def get_neuron_directions_change(self):
        """Return the lengths of the directional gradients of the hidden layer neurons."""
        with torch.no_grad():
            neuron_direction_gradient = (
                self.hidden.weight.grad
                + (
                    self.get_neuron_lengths_change()[:, None]
                    * self.get_neuron_directions()
                )
            ) / self.get_neuron_lengths()[:, None]
            return torch.linalg.norm(neuron_direction_gradient, dim=1)

    def get_alignment(self, negative_group=False):
        """Return the average and maximum angle
        between neurons in the positive or negative group, respectively."""
        with torch.no_grad():
            group = self.negative_group if negative_group else self.positive_group

            directions = self.get_neuron_directions()[group]
            angles = torch.rad2deg(
                torch.acos(torch.clamp(torch.matmul(directions, directions.T), -1, 1))
            )
            number_of_rows = angles.size(dim=0)
            number_of_pairs = number_of_rows * (number_of_rows - 1)

            # return the average and maximum angle
            return torch.sum(angles) / number_of_pairs, torch.max(angles)

    def get_total_effect(self):
        """Return the total effect of the network,
        but restricted to neurons in the positive group."""
        with torch.no_grad():
            neurons = self.hidden.weight[self.positive_group]
            coefficients = self.predict.weight.squeeze()[self.positive_group]
            return torch.sum(neurons.T * coefficients, dim=1)

    def get_sum_of_squared_weights(self):
        """Return the sum of the square of the weights in the network."""
        with torch.no_grad():
            return (
                torch.square(self.hidden.weight).sum()
                + torch.square(self.predict.weight).sum()
            )


class Experiment:
    def __init__(
        self,
        init_scale,
        width,
        d,
        n,
        learning_rate,
        training_points_noise_scale,
        uncentered_training_points=False,
        two_teachers=False,
        cutoff=20000000,
        loss_target=1e-9,
        output_dir="./",
        save_to=None,
        load_from=None,
    ):
        self.init_scale = init_scale
        self.width = width
        self.d = d
        self.n = n
        self.learning_rate = learning_rate
        self.training_points_noise_scale = training_points_noise_scale
        self.uncentered_training_points = uncentered_training_points
        self.two_teachers = two_teachers
        self.cutoff = cutoff
        self.loss_target = loss_target
        self.output_dir = output_dir
        self.metrics = []
        self.loaded_from = load_from

        loaded_state = None
        if load_from is not None:
            data = torch.load(f"{load_from}-data.pt")
            self.n = data["n"]
            self.center = data["center"]
            self.large_angle_point_count = data["large_angle_point_count"]
            self.smallest_product = data["smallest_product"]
            self.eigenvalues = data["eigenvalues"]
            self.training_points_noise_scale = data["training_points_noise_scale"]
            self.uncentered_training_points = data["uncentered_training_points"]
            self.two_teachers = data["two_teachers"]

            device_data = torch.load(f"{load_from}-devicedata.pt", map_location=device)
            self.X = device_data["X"]
            self.Y = device_data["Y"]
            self.Xval = device_data["Xval"]
            self.Yval = device_data["Yval"]
            self.Xval_out_dist = device_data["Xval outside dist"]
            self.Yval_out_dist = device_data["Yval outside dist"]
            self.teacher = device_data["teacher"]
            if self.two_teachers:
                self.teacher2 = device_data["teacher2"]
            self.transform = device_data["transform"]
            self.early_alignment_target = device_data["early_alignment_target"]

            loaded_state = torch.load(f"{load_from}-weights.pth", map_location=device)
        else:
            self.generate_training_inputs(self.uncentered_training_points)

        self.network = Net(self.d, self.width, self.X, loaded_state).to(device)

        # if save_to is not None, save the initial state of the network,
        # data points, teacher, etc. to files and exit
        if save_to is not None:
            torch.save(self.network.state_dict(), f"{save_to}-weights.pth")
            device_data = {
                "X": self.X,
                "Y": self.Y,
                "Xval": self.Xval,
                "Yval": self.Yval,
                "Xval outside dist": self.Xval_out_dist,
                "Yval outside dist": self.Yval_out_dist,
                "teacher": self.teacher,
                "transform": self.transform,
                "early_alignment_target": self.early_alignment_target,
            }
            if self.two_teachers:
                device_data["teacher2"] = self.teacher2
            torch.save(device_data, f"{save_to}-devicedata.pt")
            torch.save(
                {
                    "n": self.n,
                    "center": self.center,
                    "large_angle_point_count": self.large_angle_point_count,
                    "smallest_product": self.smallest_product,
                    "eigenvalues": self.eigenvalues,
                    "training_points_noise_scale": self.training_points_noise_scale,
                    "uncentered_training_points": self.uncentered_training_points,
                    "two_teachers": self.two_teachers,
                },
                f"{save_to}-data.pt",
            )
            sys.exit()

        self.reinitalize_network(self.init_scale)

    def generate_training_inputs(self, uncentered=False):
        """Generate training and validation data points and labels."""

        # generate a random center pointer with unit norm
        self.center = torch.nn.functional.normalize(torch.randn(self.d), dim=0)

        while True:
            # generate n training points by adding Gaussian noise to the center point
            X = self.center + torch.randn(self.n, self.d) * torch.tensor(
                self.training_points_noise_scale
            ) / torch.sqrt(torch.tensor(self.d))

            if self.two_teachers:
                X2 = -self.center + torch.randn(self.n, self.d) * torch.tensor(
                    self.training_points_noise_scale
                ) / torch.sqrt(torch.tensor(self.d))
                X = torch.cat((X, X2), 0)

            # generate n validation points by adding Gaussian noise to the center point
            Xval = self.center + torch.randn(self.n, self.d) * torch.tensor(
                self.training_points_noise_scale
            ) / torch.sqrt(torch.tensor(self.d))

            if self.two_teachers:
                Xval2 = -self.center + torch.randn(self.n, self.d) * torch.tensor(
                    self.training_points_noise_scale
                ) / torch.sqrt(torch.tensor(self.d))
                Xval = torch.cat((Xval, Xval2), 0)

            # Generate three additional sets of validation points that are all
            # drawn from a standard multivariate Gaussian distribution.
            # The sizes of the three sets are n/4, n, and 4n respectively, where
            # n is the number of training points.
            Xval_out_dist = [
                torch.randn(count, self.d)
                for count in [math.ceil(self.n / 4), self.n, 4 * self.n]
            ]

            # if we generate uncentered training points, generate the teacher
            # by adding Gaussian noise to the center point and normalize the result
            if uncentered:
                self.teacher = torch.nn.functional.normalize(
                    self.center
                    + torch.randn(self.d)
                    * torch.tensor(self.training_points_noise_scale)
                    / torch.sqrt(torch.tensor(self.d)),
                    dim=0,
                )

                if self.two_teachers:
                    self.teacher2 = torch.nn.functional.normalize(
                        -self.center
                        + torch.randn(self.d)
                        * torch.tensor(self.training_points_noise_scale)
                        / torch.sqrt(torch.tensor(self.d)),
                        dim=0,
                    )
            else:
                # otherwise, the center point is the teacher
                self.teacher = self.center

                if self.two_teachers:
                    self.teacher2 = -self.center

            if self.two_teachers:
                self.teacher = self.teacher * 1
                self.teacher2 = self.teacher2 * 3

            # find the inner product between normalized data points and the teacher neuron
            # and store some stats about them
            inner_products = torch.matmul(
                torch.nn.functional.normalize(X, dim=1), self.teacher
            )
            self.large_angle_point_count = torch.sum(inner_products < math.sqrt(1 / 2))
            self.smallest_product = torch.min(inner_products)

            # labels for the data points based on the teacher neuron(s)
            Y = torch.relu(torch.matmul(X, self.teacher))
            if self.two_teachers:
                Y = Y - torch.relu(torch.matmul(X, self.teacher2))
            Yval = torch.relu(torch.matmul(Xval, self.teacher))
            if self.two_teachers:
                Yval = Yval - torch.relu(torch.matmul(Xval, self.teacher2))
            if self.two_teachers:
                Yval_out_dist = [
                    torch.relu(torch.matmul(points, self.teacher))
                    - torch.relu(torch.matmul(points, self.teacher2))
                    for points in Xval_out_dist
                ]
            else:
                Yval_out_dist = [
                    torch.relu(torch.matmul(points, self.teacher))
                    for points in Xval_out_dist
                ]

            # Repeat the point generation process if all training labels are zero
            # otherwise, we are done and we leave the loop
            if torch.all(Y != 0):
                break

        self.eigenvalues, Q = torch.linalg.eigh(torch.matmul(X.T, X))

        # basis transform
        self.transform = (Q * torch.sign(Q.T @ self.teacher)).T

        # calculate the weighted sum of training points
        self.early_alignment_target = torch.matmul(Y, X).to(device)

        self.X = X.to(device)
        self.Xval = Xval.to(device)
        self.Xval_out_dist = [points.to(device) for points in Xval_out_dist]
        self.Y = Y.to(device)
        self.Yval = Yval.to(device)
        self.Yval_out_dist = [labels.to(device) for labels in Yval_out_dist]
        self.teacher = self.teacher.to(device)
        if self.two_teachers:
            self.teacher2 = self.teacher2.to(device)
        self.transform = self.transform.to(device)

    def reinitalize_network(self, init_scale):
        """Reinitialize the network with a different scale."""
        self.init_scale = init_scale
        self.network.reset(init_scale)
        if wandb.run is not None:
            wandb.config.update(
                self.get_experiment_parameters(include_points=(self.n * self.d <= 400))
            )

    def train(self):
        """Train the network and log a number of metrics in specific intervals."""
        self.metrics = []

        optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)

        # we regularly track a number of metrics
        # this is done at specific iterations which are choosen by a geometric series
        # it is adjusted in such a way that up to iteration 20 million, we log the metrics 1000 times
        plotting_intervals = [0] + [
            math.floor(math.pow(20000000, i / 999))
            for i in range(
                math.ceil(999 * math.log(self.cutoff) / math.log(20000000)) + 1
            )
        ]
        plotting_intervals[-1] = self.cutoff
        plotting_intervals = np.diff(plotting_intervals)
        plotting_intervals = plotting_intervals[plotting_intervals != 0]

        # we keep track of the weights at the previous logging point
        # to detect when weights are no longer changing at all and we can stop training early
        old_state = copy.deepcopy(self.network.state_dict())

        # cumulative iteration count
        i = 0

        for iterations in plotting_intervals:
            for _ in range(iterations):
                optimizer.zero_grad()
                prediction = self.network(self.X)
                loss = torch.nn.functional.mse_loss(prediction.squeeze(-1), self.Y)
                loss.backward()
                optimizer.step()
                i += 1
                # if loss drops below target value, stop training
                if loss < self.loss_target:
                    break

            # collect some statistics
            length_gradient = self.network.get_neuron_lengths_change()
            direction_gradient = self.network.get_neuron_directions_change()
            combined_neuron = self.network.get_total_effect()
            difference_to_target = torch.matmul(
                self.transform, self.teacher - combined_neuron
            )

            self.metrics.append(
                {
                    **{
                        "iteration": i,
                        "loss": loss.item(),
                        "min length grad (+)": length_gradient[
                            self.network.positive_group
                        ]
                        .min()
                        .item(),
                        "max length grad (+)": length_gradient[
                            self.network.positive_group
                        ]
                        .max()
                        .item(),
                        "mean length grad (+)": length_gradient[
                            self.network.positive_group
                        ]
                        .mean()
                        .item(),
                        "min length grad (-)": length_gradient[
                            self.network.negative_group
                        ]
                        .min()
                        .item(),
                        "max length grad (-)": length_gradient[
                            self.network.negative_group
                        ]
                        .max()
                        .item(),
                        "mean length grad (-)": length_gradient[
                            self.network.negative_group
                        ]
                        .mean()
                        .item(),
                        "min direction grad norm (+)": direction_gradient[
                            self.network.positive_group
                        ]
                        .min()
                        .item(),
                        "max direction grad norm (+)": direction_gradient[
                            self.network.positive_group
                        ]
                        .max()
                        .item(),
                        "mean direction grad norm (+)": direction_gradient[
                            self.network.positive_group
                        ]
                        .mean()
                        .item(),
                        "min direction grad norm (-)": direction_gradient[
                            self.network.negative_group
                        ]
                        .min()
                        .item(),
                        "max direction grad norm (-)": direction_gradient[
                            self.network.negative_group
                        ]
                        .max()
                        .item(),
                        "mean direction grad norm (-)": direction_gradient[
                            self.network.negative_group
                        ]
                        .mean()
                        .item(),
                        "average angle between positive neurons": self.network.get_alignment()[
                            0
                        ].item(),
                        "maximum angle between positive neurons": self.network.get_alignment()[
                            1
                        ].item(),
                        "average angle between negative neurons": self.network.get_alignment(
                            True
                        )[
                            0
                        ].item(),
                        "maximum angle between negative neurons": self.network.get_alignment(
                            True
                        )[
                            1
                        ].item(),
                        "norm of combined neuron": torch.linalg.vector_norm(
                            combined_neuron
                        ).item(),
                        "distance to teacher": torch.linalg.vector_norm(
                            self.teacher - combined_neuron
                        ).item(),
                        "angle between combined neuron and early direction target": torch.rad2deg(
                            torch.acos(
                                torch.dot(combined_neuron, self.early_alignment_target)
                                / (
                                    torch.linalg.vector_norm(combined_neuron)
                                    * torch.linalg.vector_norm(
                                        self.early_alignment_target
                                    )
                                )
                            )
                        ).item(),
                        "angle between combined neuron and teacher neuron": torch.rad2deg(
                            torch.acos(
                                torch.dot(combined_neuron, self.teacher)
                                / (
                                    torch.linalg.vector_norm(combined_neuron)
                                    * torch.linalg.vector_norm(self.teacher)
                                )
                            )
                        ).item(),
                        "sum of squared weights": self.network.get_sum_of_squared_weights().item(),
                        "nuclear norm": torch.linalg.matrix_norm(
                            self.network.hidden.weight, ord="nuc"
                        ).item(),
                        "nonzero predictions": torch.count_nonzero(prediction).item(),
                        "average num of points seen (+)": torch.sum(
                            (
                                self.X
                                @ self.network.hidden.weight[
                                    self.network.positive_group
                                ].T
                            )
                            > 0
                        )
                        / torch.sum(self.network.positive_group),
                        "val loss": torch.nn.functional.mse_loss(
                            self.network(self.Xval).squeeze(-1), self.Yval
                        ).item(),
                    },
                    **{
                        f"val loss outside dist {points.size(dim=0)}": torch.nn.functional.mse_loss(
                            self.network(points).squeeze(-1), labels
                        ).item()
                        for points, labels in zip(
                            self.Xval_out_dist, self.Yval_out_dist
                        )
                    },
                    **{
                        f"difference to target in dim {j+1}": difference_to_target[
                            self.d - j - 1
                        ].item()
                        for j in range(self.d)
                    },
                }
            )

            # also log these to W&B, but exclude the difference to target metric
            # because there will be too many of them if running for large dimensions
            wandblogging = dict(self.metrics[-1])
            for j in range(self.d):
                del wandblogging[f"difference to target in dim {j+1}"]
            wandb.log(wandblogging)

            # stop training, if loss has reached the target value
            # or if the hidden weights are no longer changing
            if loss < self.loss_target or torch.equal(
                old_state["hidden.weight"], self.network.hidden.weight
            ):
                break

            old_state = copy.deepcopy(self.network.state_dict())

    def generate_filename(self):
        """Generate output filename based on some experiment parameters and the time."""
        now = datetime.now()
        return f"{self.d}_{self.n}_{self.width}_{self.init_scale}_{self.training_points_noise_scale}_{self.uncentered_training_points}_{self.two_teachers}_{self.learning_rate}_{now.strftime('%y%m%d%H%M%S')}_{self.network.get_initialization_hash()}.csv"

    def get_experiment_parameters(self, include_points=True, include_weights=False):
        result = {
            "d": self.d,
            "n": self.n,
            "width": self.width,
            "init scale": self.init_scale,
            "initialization hash": self.network.get_initialization_hash(),
            "loaded from": self.loaded_from,
            "learning rate": self.learning_rate,
            "training points noise scale": self.training_points_noise_scale,
            "uncentered training points": self.uncentered_training_points,
            "two teachers": self.two_teachers,
            "cutoff": self.cutoff,
            "loss target": self.loss_target,
            "maximum angle to teacher": torch.rad2deg(
                torch.acos(self.smallest_product)
            ).item(),
            "large angle point count": self.large_angle_point_count.item(),
            "eigenvalues": self.eigenvalues.tolist(),
            "teacher neuron": self.teacher.tolist(),
            "center": self.center.tolist(),
            "always dead": self.network.always_dead.tolist(),
            "positive group": self.network.positive_group.tolist(),
            "negative group": self.network.negative_group.tolist(),
        }
        if self.two_teachers:
            result["teacher neuron 2"] = self.teacher2.tolist()
        if include_points:
            result["data points"] = self.X.tolist()
        if include_weights:
            result["init weights"] = self.network._initial_state[
                "hidden.weight"
            ].tolist()
            result["weights"] = self.network.hidden.weight.tolist()
            result["predict weights"] = self.network.predict.weight.tolist()
        return result

    def write_results(self, filename=None):
        """Write experiment results to file. If filename is not given, generate one.
        The file will be a csv file with a header containing the experiment parameters.
        """
        if filename is None:
            filename = self.generate_filename()

        experiment_parameters = self.get_experiment_parameters(include_weights=True)

        parameters_string = (
            "".join(
                [
                    f"# {line}"
                    for line in json.dumps(experiment_parameters, indent=4).splitlines(
                        True
                    )
                ]
            )
            + "\n"
        )
        with open(os.path.join(self.output_dir, filename), "w", encoding="utf-8") as f:
            f.write(parameters_string)
        pd.DataFrame(self.metrics).to_csv(
            os.path.join(self.output_dir, filename), mode="a", index=False
        )


def directory(path):
    if os.path.isdir(path):
        return os.path.abspath(path)
    else:
        raise NotADirectoryError(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())],
        default="cpu",
    )

    parser.add_argument("--width", type=int, default=25)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--training-points-noise-scale", type=float)
    parser.add_argument("--uncentered-training-points", action="store_true")
    parser.add_argument("--two-teachers", action="store_true")
    parser.add_argument("--init-scales", nargs="+", type=float, default=[1.0])
    parser.add_argument("--cutoff", type=int, default=20000000)
    parser.add_argument("--loss-target", type=float, default=1e-9)
    parser.add_argument("--output-dir", type=directory, default="./")
    parser.add_argument("--save-to", type=str, default=None)
    parser.add_argument("--load-from", type=str, default=None)

    args = parser.parse_args()

    # The default value for noise scaling is choosen differently for both input generation
    # schemas, such that the expected angle (for large dimensions) is roughly 45 degrees in either case
    if args.training_points_noise_scale is None:
        args.training_points_noise_scale = (
            math.sqrt(math.sqrt(2) - 1) if args.uncentered_training_points else 1.0
        )
    device = args.device

    test = None
    for init_scale in args.init_scales:
        if args.save_to is None:
            wandb.init(
                mode="online",
                project="dummy",
                config={"device": device},
            )

        if test is None:
            test = Experiment(
                init_scale,
                args.width,
                args.d,
                args.n,
                args.learning_rate,
                args.training_points_noise_scale,
                args.uncentered_training_points,
                args.two_teachers,
                args.cutoff,
                args.loss_target,
                args.output_dir,
                args.save_to,
                args.load_from,
            )
        else:
            test.reinitalize_network(init_scale)

        # train and make sure that if the training stops unexpectedly,
        # the results up to that point are still saved
        atexit.register(test.write_results)
        test.train()
        atexit.unregister(test.write_results)

        # save result once training finished normally
        test.write_results()
        wandb.finish()
