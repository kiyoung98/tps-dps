import wandb
from .plot import Plot
from .metrics import Metric


class Log:
    def __init__(self, args, sys):
        self.wandb = args.wandb
        self.save_dir = args.save_dir
        self.save_freq = args.save_freq

        self.plot = Plot(args, sys)
        self.metric = Metric(args, sys)

    def sample(self, dps, rollout, positions, potentials):
        log = {"log_z": dps.params["log_z"]}
        metrics = self.metric(positions, potentials)
        log.update(metrics)

        # TODO: checkpoint code 짜기
        if rollout % self.save_freq == 0:
            plots = self.plot(positions, rollout)
            log.update(plots)

        if self.wandb:
            wandb.log(log, step=rollout)

        print(f"log_z: {log['log_z']}")
        print(f"rmsd: {log['rmsd']} ± {log['rmsd_std']}")
        print(f"thp: {log['thp']}")
        print(f"etp: {log['etp']} ± {log['etp_std']}")

    def train(self, rollout, loss, sampling_time, training_time):
        if self.wandb:
            wandb.log({"loss": loss}, step=rollout)
            wandb.log({"sampling_time": sampling_time}, step=rollout)
            wandb.log({"training_time": training_time}, step=rollout)

        print(f"Loss: {loss}")
        print(f"Sampling time: {sampling_time}")
        print(f"Training time: {training_time}")
