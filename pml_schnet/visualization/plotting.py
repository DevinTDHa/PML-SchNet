import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem.rdchem import GetPeriodicTable

periodic_table = GetPeriodicTable()


def an_to_element(a_num):
    return periodic_table.GetElementSymbol(a_num)


def show(
    x: dict,
    outfile=None,
    energy_key="energy",
    forces_key="forces",
    figsize=(10, 10),
    lim=4,
):
    sns.set_theme()

    data = x

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    ax.set_xlim((-lim, lim))
    ax.set_ylim((-lim, lim))
    ax.set_zlim((-lim, lim))

    xs = data["_positions"][:, 0]
    ys = data["_positions"][:, 1]
    zs = data["_positions"][:, 2]

    energy = data[energy_key].item()

    # Plot points
    ax.scatter(
        xs=xs,
        ys=ys,
        zs=zs,
        color="black",
        label=f"Atoms E={energy:6.4f}",
        s=200,
        edgecolors="grey",
        linewidths=3,
    )

    # Plot Atom Texts
    for number, pos in zip(data["_atomic_numbers"], data["_positions"]):
        ax.text(
            s=an_to_element(number.item()), x=pos[0], y=pos[1] + 0.2, z=pos[2], size=16
        )

    if forces_key:
        # Plot Force Vectors
        forces_xs = data[forces_key][:, 0]
        forces_ys = data[forces_key][:, 1]
        forces_zs = data[forces_key][:, 2]
        if (forces_xs + forces_ys + forces_zs).sum() != 0:
            f_scaling = 0.05
            ax.quiver(
                xs,
                ys,
                zs,
                forces_xs,
                forces_ys,
                forces_zs,
                color="red",
                length=f_scaling,
                label="Force",
            )

            plt.title(f"Energy and Forces (F scaled to {f_scaling * 100}%)")
    else:
        plt.title(f"Energy and Atom Positions")

    plt.legend()

    if outfile:
        fig.savefig(outfile)

    return fig



def plot_loss(losses):
    return
    fig = px.line(pd.DataFrame(losses), x="epoch", y="loss", title="Loss over epoch")
    fig.show()


