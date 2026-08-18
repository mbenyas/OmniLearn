import numpy as np
import h5py as h5
import os
from optparse import OptionParser

# to be run before preprocess_lhco.py, so that the outputs of fastjet_awkward.ipynb
# are compatible with preprocess_lhco.py

# Converts the fastjet-clustered LHCO output (jet_data/constituents/mask, absolute
# pt/eta/phi) from fastjet_awkward.ipynb into the data/jet format expected by
# utils.LHCODataLoader (used by preprocess_lhco.py): 'data' holds the standard
# 7-feature PET particle representation, 'jet' holds [pt, eta, phi, mass, multiplicity].

def convert_chunk(jet, parts, mask):
    # jet/parts store [pt, eta, phi, mass] / [pt, eta, phi]; energy isn't stored,
    # so recover it assuming on-shell 4-vectors: E = sqrt(m^2 + (pt*cosh(eta))^2).
    # Constituents are treated as massless (mass=0) per the standard PET convention.
    jet_pt, jet_eta, jet_phi, jet_mass = [jet[..., i] for i in range(4)]
    jet_e = np.sqrt(jet_mass**2 + (jet_pt*np.cosh(jet_eta))**2)

    part_pt, part_eta, part_phi = [parts[..., i] for i in range(3)]
    part_e = part_pt*np.cosh(part_eta)

    delta_phi = part_phi - jet_phi[:, :, None]
    delta_phi = np.where(delta_phi > np.pi, delta_phi - 2*np.pi, delta_phi)
    delta_phi = np.where(delta_phi < -np.pi, delta_phi + 2*np.pi, delta_phi)

    # 7 PET particle features, in the order LHCODataLoader's mean_part/std_part expect:
    # [eta_rel, phi_rel, log(1-pT_rel), log(pT), log(1-E_rel), log(E), deltaR]
    points = np.zeros(part_pt.shape + (7,), dtype=np.float32)
    points[..., 0] = part_eta - jet_eta[:, :, None]
    points[..., 1] = delta_phi
    # np.ma.log masks non-finite results (e.g. log(0) for zero-padded particles,
    # or log of a negative value if a constituent's pt exceeds the jet's vector-sum pt)
    # and .filled(0) zeroes them out instead of propagating NaN/-inf.
    points[..., 2] = np.ma.log(1.0 - part_pt/jet_pt[:, :, None]).filled(0)
    points[..., 3] = np.ma.log(part_pt).filled(0)
    points[..., 4] = np.ma.log(1.0 - part_e/jet_e[:, :, None]).filled(0)
    points[..., 5] = np.ma.log(part_e).filled(0)
    points[..., 6] = np.hypot(points[..., 0], points[..., 1])
    points *= mask[..., None]  # re-zero padded particles (log(1)=0 already, but not log(pT))

    mult = mask.sum(axis=-1)
    # jet feature order [pt, eta, phi, mass, multiplicity] matches LHCODataLoader's
    # mean_jet/std_jet and its preprocess_jet phi transform (index 2).
    jet_out = np.stack([jet_pt, jet_eta, jet_phi, jet_mass, mult], axis=-1).astype(np.float32)
    return points, jet_out


def convert(input_path, output_path, chunk_size=20000):
    # Full arrays don't fit in memory at float64 (background is 1M events x 2 x 268
    # particles x 7 features -> ~30GB), so read/write in chunks at float32 instead.
    with h5.File(input_path, 'r') as fin:
        n_events, n_jets, n_part, _ = fin['constituents'].shape

        with h5.File(output_path, 'w') as fout:
            data_ds = fout.create_dataset('data', shape=(n_events, n_jets, n_part, 7), dtype=np.float32)
            jet_ds = fout.create_dataset('jet', shape=(n_events, n_jets, 5), dtype=np.float32)

            for start in range(0, n_events, chunk_size):
                end = min(start + chunk_size, n_events)
                jet = fin['jet_data'][start:end]
                parts = fin['constituents'][start:end]
                mask = fin['mask'][start:end, :, :, 0].astype(bool)

                points, jet_out = convert_chunk(jet, parts, mask)
                data_ds[start:end] = points
                jet_ds[start:end] = jet_out


if __name__ == '__main__':
    parser = OptionParser(usage="%prog [opt]")
    parser.add_option("--folder", type="string", default="/global/cfs/cdirs/m3246/mbenyas/OmniLearn",
                       help="Folder containing the LHCO subfolder with the fastjet notebook outputs")

    (flags, args) = parser.parse_args()

    for sample in ['background', 'signal']:
        convert(
            os.path.join(flags.folder, 'LHCO', 'processed_data_{}.h5'.format(sample)),
            os.path.join(flags.folder, 'LHCO', 'processed_data_{}_rel_new.h5'.format(sample)),
        )
