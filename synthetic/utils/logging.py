import os
import sys
import logging
from flax.training import checkpoints

from .plot import Plot
from .metrics import Metric


class Log:
    def __init__(self, args, mds):
        self.save_dir = args.save_dir

        self.plot = Plot(args, mds)
        self.metric = Metric(args, mds)

        self.rmsd = float("inf")

        # Logger basic configurations
        self.logger = logging.getLogger("tps")
        self.logger.setLevel(logging.INFO)

        # File handler
        if args.training:
            log_file = "train.log"
        else:
            log_file = "eval.log"
        log_file = os.path.join(args.save_dir, log_file)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

        for k, v in vars(args).items():
            self.logger.info(f"{k}: {v}")

    def info(self, message):
        if self.logger:
            self.logger.info(message)

    def sample(self, rollout, policy, positions, potentials):
        self.logger.info("-----------------------------------------------------------")
        self.logger.info(f"Rollout: {rollout}")

        metrics = self.metric(positions, potentials)

        # if rollout % 10 == 0:
        #     self.plot(positions)
        self.plot(positions, rollout)
        #     checkpoints.save_checkpoint(
        #         self.save_dir, policy, rollout, prefix="policies/"
        #     )

        # if self.rmsd > metrics["rmsd"]:
        #     self.rmsd = metrics["rmsd"]
        #     checkpoints.save_checkpoint(self.save_dir, policy, 0, prefix="rmsd_policy")

        # self.logger.info(f"log_z: {policy.log_z}")
        self.logger.info(f"rmsd: {metrics['rmsd']} ± {metrics['rmsd_std']}")
        self.logger.info(f"thp: {metrics['thp']}")
        self.logger.info(f"etp: {metrics['etp']} ± {metrics['etp_std']}")
