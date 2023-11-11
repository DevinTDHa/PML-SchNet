import os
import shutil
import tempfile

from schnetpack.transform import ASENeighborList
from tqdm import tqdm
import numpy as np
from ase.db import connect
from schnetpack.datasets import ISO17


def fix_iso_17_db(data_path="./iso17.db"):
    if "iso17_fixed" not in os.listdir():
        iso17data = ISO17(
            "./iso17.db",
            fold="reference_eq",
            batch_size=1,
            num_train=1,
            num_test=1,
            num_val=1,
            transforms=[ASENeighborList(cutoff=5.0)],
        )

        iso17data.prepare_data()
        iso17data.setup()
        # fix databases
        tmpdir = tempfile.mkdtemp("iso17")
        for fold in ISO17.existing_folds:
            dbpath = os.path.join(data_path, "iso17", fold + ".db")
            tmp_dbpath = os.path.join(tmpdir, "tmp.db")
            with connect(dbpath) as conn:
                with connect(tmp_dbpath) as tmp_conn:
                    tmp_conn.metadata = {
                        "_property_unit_dict": {
                            ISO17.energy: "eV",
                            ISO17.forces: "eV/Ang",
                        },
                        "_distance_unit": "Ang",
                        "atomrefs": {},
                    }
                    # add energy to data dict in db
                    for idx in tqdm(
                        range(len(conn)), f"parsing database file {dbpath}"
                    ):
                        atmsrw = conn.get(idx + 1)
                        data = atmsrw.data
                        data[ISO17.forces] = np.array(data[ISO17.forces])
                        data[ISO17.energy] = np.array([atmsrw.total_energy])
                        tmp_conn.write(atmsrw.toatoms(), data=data)

            os.remove(dbpath)
            os.rename(tmp_dbpath, dbpath)
        shutil.rmtree(tmpdir)
        os.system("touch iso17_fixed")
