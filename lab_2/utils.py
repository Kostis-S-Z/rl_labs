import pylab


def plot_data(episodes, scores, max_q_mean, model_name):
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
