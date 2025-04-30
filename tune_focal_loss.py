import os
import keras_tuner as kt
from birdseye_yolo_trainer import BirdsEyeTrainer
from config_birdseye import get_config

class BirdsEyeHyperModel(kt.HyperModel):
    def __init__(self, base_config):
        self.base_config = base_config

    def build(self, hp):
        config = self.base_config.copy()
        config["alpha"] = hp.Float("alpha", min_value=0.1, max_value=0.9, step=0.2)
        config["gamma"] = hp.Float("gamma", min_value=0.5, max_value=5.0, step=0.5)
        config["filename"] += f"_a{int(config['alpha']*100)}_g{int(config['gamma']*10)}"

        trainer = BirdsEyeTrainer(config)
        trainer.setup_data()
        trainer.build_model()
        return trainer.model

    def fit(self, hp, model, *args, **kwargs):
        config = self.base_config.copy()
        config["alpha"] = hp.get("alpha")
        config["gamma"] = hp.get("gamma")
        config["filename"] += f"_a{int(config['alpha']*100)}_g{int(config['gamma']*10)}"

        trainer = BirdsEyeTrainer(config)
        trainer.run()
        return trainer.model.history

if __name__ == "__main__":
    base_config = get_config(mode="tile")  # only tile mode uses focal loss

    tuner = kt.RandomSearch(
        hypermodel=BirdsEyeHyperModel(base_config),
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=10,
        executions_per_trial=1,
        directory="focal_loss_tuning",
        project_name="birdseye"
    )

    tuner.search()
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print(f"Best alpha: {best_hps.get('alpha')}")
    print(f"Best gamma: {best_hps.get('gamma')}")
