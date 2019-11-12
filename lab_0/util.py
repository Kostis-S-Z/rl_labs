import matplotlib.pyplot as plt


def plot(maze):

    fig, ax = plt.subplots(1)

    ax.plot(maze)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_ylim(bottom=0, top=5)
    ax.set_xlim(left=0, right=5)
    ax.grid(b=True)
    plt.show()
