import wandb


def init_wandb_run(model_name, version_name):
    wandb.login()
    return wandb.init(
        entity='asmazurik-company',
        project=f"capstone_train_{model_name}",
        name=f"{version_name}_{model_name}_regression",
        config={
            "model": f"{model_name} regression (sklearn)",
            "version": version_name,
        },
    )
