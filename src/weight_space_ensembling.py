import collections


def weight_space_ensembling(model_finetuned, model_zeroshot, alpha=0.5):
    """
    Idea from:
        Paper: https://arxiv.org/pdf/2109.01903.pdf
        Github :https://github.com/mlfoundations/wise-ft
        alpha=0: only finetuned model
        alpha=1: only zeroshot model
    """
    theta_0 = model_zeroshot.state_dict()
    theta_1 = model_finetuned.state_dict()

    # make sure checkpoints are compatible
    assert set(theta_0.keys()) == set(theta_1.keys())

    # interpolate between checkpoints with mixing coefficient alpha
    theta = {
        key: (1 - alpha) * theta_0[key] + alpha * theta_1[key] for key in theta_0.keys()
    }
    # theta = collections.OrderedDict(theta)
    # update the model acccording to the new weights
    return theta
