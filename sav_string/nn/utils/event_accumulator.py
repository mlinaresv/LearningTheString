import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_tensorboard_scalars(log_dirs, tags=None, labels=None, colors=None):
    """
    Plot scalar data from multiple TensorBoard runs.

    Args:
        log_dirs (list of str): List of log directories (one per run).
        tags (list, optional): List of scalar tags to plot. If None, uses common tags.
        labels (list, optional): Labels for each run (for legend).
    """

    # Load all event accumulators
    eas = []
    for log_dir in log_dirs:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        eas.append(ea)

    # Get available tags per run
    all_tags = [set(ea.Tags()['scalars']) for ea in eas]

    # Find common tags if none specified
    if tags is None:
        tags = list(set.intersection(*all_tags))

    if not tags:
        print("No common scalar tags found.")
        return

    print("Tags being plotted:", tags)

    # Default labels
    if labels is None:
        labels = [f"run_{i}" for i in range(len(log_dirs))]

    # Create subplots
    n_tags = len(tags)
    fig, axes = plt.subplots(n_tags, 1, figsize=(6, 4 * n_tags), squeeze=False)

    for i, tag in enumerate(tags):
        ax = axes[i][0]

        for ea, label, color in zip(eas, labels, colors):
            available_tags = ea.Tags()['scalars']

            if tag not in available_tags:
                print(f"Tag '{tag}' not found in one run, skipping that run.")
                continue

            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            ax.plot(steps, values, label=label, color=color)

        ax.set_title(tag)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 1e1)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()

    # Save figure
    filename = "_".join([t.replace('/', '_') for t in tags])
    plt.savefig(f"{filename}.pdf")
    plt.show()


if __name__ == '__main__':

    log_dirs = [
        '/u/anasynth/linares/LearningTheString/sav_string/runs/20260403_082938',
        '/u/anasynth/linares/LearningTheString/sav_string/runs/20260403_105543',
        '/u/anasynth/linares/LearningTheString/sav_string/runs/20260408_095659',
        '/u/anasynth/linares/LearningTheString/sav_string/runs/20260407_122744',
        #'/u/anasynth/linares/LearningTheString/sav_string/runs/20260407_131319'
    ]
    tags = [
        'Gradients/EstimatorModel',
        'Gradients/DecoderModel',
        #'Gradients/NonPhysicsModelModel',
        'Train/Loss',
        'Validation/InferenceLoss(MSE)',
        'Validation/ObservationError(MSE)'
    ]
    labels = [
        'encodec-mse',
        'encodec-mae',
        'encodec-mse-reconstruction-loss',
        'mlp_800k',
        #'mlp_290k'
    ]

    plot_tensorboard_scalars(log_dirs, tags, labels)