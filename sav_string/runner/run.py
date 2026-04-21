import sys
sys.path.append('/u/anasynth/linares/LearningTheString')
from sav_string.train import train
from sav_string.sav_solver import SAVSolver
from sav_string.utils.loader import load_optimizer
from sav_string.utils.losses import define_loss_function
from sav_string.utils.loader import load_datasets, load_learnable_objects



def main(config: dict, n_traj: int, log_dir: str, device: str):


    # DEFINE THE DATASETS AND DATALOADERS.
    print('--> Building the training and validation dataset...')

    train_dataset, val_dataset = load_datasets(config, n_traj)

    
    # DEFINE THE LEARNABLE OBJECTS.
    model, estimator, decoder = load_learnable_objects(
        config=config,
        dataset=train_dataset,
        device=device,
        load_model_dictionary=config["checkpoint"]["load_dictionary"],
        checkpoint_file=config["checkpoint"]["checkpoint_file"]
    )


    # DEFINE THE NUMERICAL SOLVER.
    solver = SAVSolver(model, sr=config["train"]["sampling_rate"], lib='torch')

    # DETERMINE THE TRAINING LOSSES.
    baseline_loss = define_loss_function(config["baseline_loss_type"], dt=solver.dt)
    reconstruction_loss = define_loss_function(config["reconstruction_loss_type"], dt=solver.dt)

    ###### OPTIMIZER ######
    optimizer = load_optimizer(
        model=model,
        estimator=estimator,
        decoder=decoder,
        lr=config["train"]["learning_rate"],
        load_optimizer_dictionary=config["optimizer"]["load_optimizer_dictionary"],
        checkpoint_file=config["optimizer"]["checkpoint_file"]
    )

    # TRAIN THE MODEL.

    train(

        sr=config["train"]["sampling_rate"],
        model=model,
        solver=solver,
        estimator=estimator,
        decoder=decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=config["train"]["training_batch_size"],
        val_batch_size=config["train"]["validation_batch_size"],
        baseline_loss_function=baseline_loss,
        reconstruction_loss_function=reconstruction_loss,
        optimizer=optimizer,
        epochs=config["train"]["epochs"],
        max_steps=config["train"]["max_steps"],
        device=device,
        log_each_n_updates=config["train"]["log_each_n_updates"],
        infer_each_n_updates=config["train"]["infer_each_n_updates"],
        log_dir=log_dir,
        test_audio_duration=config["train"]["test_audio_duration"],
        test_q_interval=config["train"]["test_q_interval"]

    )
