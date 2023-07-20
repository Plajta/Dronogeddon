import wandb

def Init(project_name, configuration, run_name = ""):
    wandb.init(
        project=project_name,
        config=configuration
    )
    if len(run_name) > 0:
        wandb.run.name = run_name

    return wandb.config
    #use the config as main config for layers!

def InitSweep(configuration, project_name, run):
    sweep_id = wandb.sweep(sweep=configuration, project=project_name)
    wandb.agent(sweep_id=sweep_id, function=run, count=4)

def End():
    wandb.finish()