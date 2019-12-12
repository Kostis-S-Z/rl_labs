import pylab
import os
import json


def plot_data(episodes, scores, max_q_mean, mean_scores, model_name):
    """
    Plots the score per episode as well as the maximum q value per episode, averaged over precollected states.
    """
    pylab.figure(0)
    pylab.plot(episodes, max_q_mean, 'b')
    pylab.xlabel("Episodes")
    pylab.ylabel("Average Q Value")
    pylab.savefig(model_name + "/q_values.png")
    pylab.show()

    pylab.figure(1)
    pylab.plot(episodes, scores, 'b')
    pylab.xlabel("Episodes")
    pylab.ylabel("Score")
    pylab.savefig(model_name + "/scores.png")
    pylab.show()

    pylab.figure(2)
    pylab.plot(episodes[99:], mean_scores[99:], 'b')
    pylab.xlabel("Episodes")
    pylab.ylabel("Mean Score of last 100 episodes")
    pylab.savefig(model_name + "/mean_scores.png")
    pylab.show()


def plot_loss(loss, model_name):
    x_axis = range(1, len(loss) + 1)
    y_axis = loss
    pylab.plot(x_axis, y_axis)
    pylab.xlabel('Epochs')
    pylab.ylabel('Loss')
    pylab.savefig(model_name + "/loss.png")
    pylab.show()


def save_params(model_name, params, net):
    if not os.path.exists(model_name):
        os.mkdir(model_name)

    with open(model_name + '/params.json', 'w') as fp:
        json.dump(params, fp)

    with open(model_name + '/net.json', 'w') as fp:
        json.dump(net, fp)
