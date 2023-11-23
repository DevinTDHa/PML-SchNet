import os.path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from schnetpack.datasets import MD17
from schnetpack.transform import ASENeighborList
from tqdm import tqdm

if __name__ == "__main__":
    sns.set_theme()

    norm = Normalize(vmin=-406757.5912640221, vmax=-406724.94098081137)
    sm = ScalarMappable(cmap="hot", norm=norm)

    md17data = MD17(
        "../md17.db",
        molecule="aspirin",
        batch_size=10,
        num_train=1000,
        num_val=10,
        transforms=[ASENeighborList(cutoff=5.0)],
    )
    md17data.prepare_data()
    md17data.setup()

    if not os.path.exists("plots"):
        os.mkdir("plots")

    for i in tqdm(range(300)):
        data = md17data.dataset[i]

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        lim = 4
        ax.set_xlim((-lim, lim))
        ax.set_ylim((-lim, lim))
        ax.set_zlim((-lim, lim))

        xs = data["_positions"][:, 0]
        ys = data["_positions"][:, 1]
        zs = data["_positions"][:, 2]

        forces_xs = data["forces"][:, 0]
        forces_ys = data["forces"][:, 1]
        forces_zs = data["forces"][:, 2]

        energy = data["energy"].item()

        # Letters next to the Atoms
        an_to_element = {1: "H", 6: "O", 8: "C"}

        # Plot Atoms
        for number, pos in zip(
            data["_atomic_numbers"].numpy(), data["_positions"].numpy()
        ):
            ax.text(
                s=an_to_element[number], x=pos[0], y=pos[1] + 0.2, z=pos[2], size=16
            )

        # Plot points
        ax.scatter(
            xs=xs,
            ys=ys,
            zs=zs,
            color=sm.to_rgba(energy),
            label=f"Atoms E={energy:6.4f}",
            s=200,
            edgecolors="grey",
            linewidths=3,
        )

        # Plot Force Vectors
        ax.quiver(
            xs,
            ys,
            zs,
            forces_xs,
            forces_ys,
            forces_zs,
            color="black",
            length=0.02,
            label="Force",
            alpha=0.5,
        )
        fig.colorbar(sm, ax=ax, label="Energy in Ha", location="bottom")
        plt.title("Energy and Force Acting on Aspirin")
        plt.legend()

        fig.savefig(f"plots/{i:06d}")
        plt.close()
