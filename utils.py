import numpy as np
from typing import Tuple
import re


def csa_count(shift_name, val):
    if shift_name is not None:
        a = shift_name.split(".")
        return "{}.{}".format(a[0], int(a[1])+val)
    else:
        return "0"


def read_from_xeasy(peak_list_path: str, permute_order_to_match_definition: bool = False):

    """
        Reads peak list from XEASY file
    """

    # Peak list file handler
    with open(peak_list_path, 'r') as f:

        # Parse header: experiment type and axes names
        spectrum_name, axes_names = None, []

        next_line = f.readline()
        while next_line.startswith('#'):
            if next_line.startswith('#SPECTRUM'):
                next_line_split = next_line.split()
                axes_names = next_line_split[2:]
                spectrum_name = next_line_split[1]
            next_line = f.readline()
        num_axes = len(axes_names)
        none_assignment = [None] * num_axes
        vc_pattern = re.compile(r"VC [0-9]+\.[0-9]+")

        if spectrum_name is None:
            raise Exception("Attempt to read xeasy peak list: unrecognized type of spectrum ({}).".format(peak_list_path))

        # Adjust order to match spectrum definition (method will fail here if axes names do not match spectrum definition)
        if permute_order_to_match_definition:
            definition_names = spectrum_ops.get_axes_names(spectrum_name)
            permutation = np.asarray([axes_names.index(e) for e in definition_names], dtype=int)
        else:
            permutation = np.asarray(range(len(axes_names)), dtype=int)

        # Parse content
        all_lines = [next_line] + f.readlines()
        peak_positions, peak_assignments, peak_intensities, peak_colors, peak_vc = [], [], [], [], []

        for line in all_lines:

            if len(line) == 0:
                continue

            tmp = line.split('#', 1)
            line_content = tmp[0]
            line_comment = tmp[1] if len(tmp) > 1 else ""
            line_parts = line_content.split()
            vc_flag = "VC " in line_comment

            if len(line_parts) < 9:

                # Read volume contributions (if available in the text file)
                if vc_flag:
                    peak_assignment = [int(e) - 1 for e in line_parts]
                    peak_assignment = [peak_assignment[e] for e in permutation]
                    matches = vc_pattern.findall(line_comment)
                    if len(matches) > 0:
                        vc = float(matches[0][3:])
                        if vc > peak_vc[-1]:
                            peak_vc[-1] = vc
                            peak_assignments[-1] = peak_assignment
                    continue

                else:
                    printw("XEASYReader -> Skipping line {}".format(line))
                    continue

            peak_positions.append(line_parts[1:(1 + num_axes)])
            peak_intensities.append(line_parts[num_axes + 3])
            peak_assignment = line_parts[(7 + num_axes):]
            peak_colors.append(line_parts[num_axes + 1])

            # Read volume contributions (if available in the text file)
            if vc_flag:
                matches = vc_pattern.findall(line_comment)
                peak_vc.append(float(matches[0][3:]) if len(matches) > 0 else 1.0)
            else:
                peak_vc.append(1.0)

            if len(peak_assignment) > 0 and all("." in e for e in peak_assignment): # Assignments in form <atom_label>.<residue_id> "HA.6"
                peak_assignment = [csa_count(e, -1).replace("Q", "H") for e in peak_assignment] # Replace Q with H to unify CYANA and BMRB
            else:
                peak_assignment = none_assignment

            peak_assignments.append(peak_assignment)

        peak_positions = np.asarray(peak_positions, dtype=float)
        peak_assignments = np.asarray(peak_assignments)
        peak_intensities = np.asarray(peak_intensities, dtype=float)
        peak_colors = np.asarray(peak_colors, dtype=int)

        if peak_positions.ndim > 1:
            peak_positions = peak_positions[:, permutation]
            peak_assignments = peak_assignments[:, permutation]

        return PeakList(peak_positions, [axes_names[e] for e in permutation], PeakList.UNIT_PPM,
                    info={
                        PeakList.ASSIGNMENT: peak_assignments,
                        PeakList.INTENSITIES: peak_intensities,
                        PeakList.VOLUME_CONTRIBUTION: peak_vc,
                        PeakList.SPECTRUM_TYPE: spectrum_name,
                        PeakList.COLOR: peak_colors,

                        # If nothing is specified in the filename, read and store colors, otherwise overwrite colors with inter or intra.
                        PeakList.INTER_MOLECULAR: np.ones_like(peak_colors, dtype=bool) if "@INTER" in peak_list_path else peak_colors == np.ones_like(peak_colors) * 9,
                        PeakList.INTRA_MOLECULAR: np.ones_like(peak_colors, dtype=bool) if "@INTRA" in peak_list_path else peak_colors == np.ones_like(peak_colors) * 8,
                    }
                )


class PeakList(object):

    # INFO LABELS
    ASSIGNMENT = "ASSIGNMENT"
    INTENSITIES = "INTENSITIES"
    RESPONSES = "RESPONSES"
    EMBEDDINGS = "EMBEDDINGS"
    INTER_MOLECULAR = "INTER_MOLECULAR"
    INTRA_MOLECULAR = "INTRA_MOLECULAR"
    VOLUME_CONTRIBUTION = "VOLUME_CONTRIBUTION"
    FILE_NAME = "FILE_NAME"
    PROP_DICT = "PROPERTY_DICTIONARY"
    SPECTRUM_TYPE = "SPECTRUM_TYPE"
    COLOR = "COLOR"

    # UNITS
    UNIT_PPM = "ppm"
    UNIT_ID = "id"
    UNIT_UNKNOWN = "na"

    # WARNINGS
    SUPPRESS_WARNINGS = False

    def __init__(self, data: np.ndarray, axes_names: Tuple[str], unit: str, info=None):
        self.data = np.asarray(data, float)
        self.axes_names = axes_names
        self.unit = unit
        self.info = {} if info is None else info
        self.dim = self.data.shape[1] if len(self.data.shape) == 2 else len(axes_names) # Handling empty peak lists

        # Iterator
        self.curr = 0

        # Data validator (discrepancy between axes names and peak list dimensionality)
        if self.dim != len(self.axes_names):
            print("Corrupted PeakList object. Data dimensionality is {}, whereas axes_names are {}".format(self.dim, self.axes_names))
            diff = self.dim - len(self.axes_names)
            if diff > 0:
                self.axes_names += ["N/A"] * abs(diff)
            else:
                self.axes_names = self.axes_names[:self.dim]

        if len(self.data) == 0 and not PeakList.SUPPRESS_WARNINGS:
            print("Peak list object contains no peaks. Dimensionality taken from axes_names")

    def shape(self):
        return self.data.shape

    def is_empty(self):
        return len(self.data) == 0


    def shift(self, vector):
        if isinstance(vector, list):
            vector = np.asarray(vector)
        self.data = self.data + np.expand_dims(vector, axis=0)

    def get(self, label):
        return self.info[label] if label in self.info else None

    def set(self, label, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)

        self.info[label] = value

    def has(self, label):
        return label in self.info

    def head(self, num_peaks):

        if num_peaks > len(self):
            return copy.deepcopy(self)

        else:
            num_peaks = int(num_peaks)
            new_data = self[:num_peaks]

            new_info = {}
            for key in self.info.keys():
                var = self.get(key)
                if isinstance(var, np.ndarray) or isinstance(var, list):
                    new_info[key] = var[:num_peaks]

            return PeakList(new_data, self.axes_names, self.unit, info=new_info)

    def subset(self, idx):
        new_data = self[idx]
        new_info = {}

        for key in self.info.keys():
            var = self.get(key)
            if isinstance(var, np.ndarray):
                new_info[key] = var[idx]
            else:
                new_info[key] = copy.deepcopy(var)

        return PeakList(new_data, self.axes_names, self.unit, info=new_info)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        self.curr = 0
        return self

    def __next__(self):
        if self.curr >= len(self):
            raise StopIteration
        else:
            self.curr += 1
            return self[self.curr - 1, :]

    def __getitem__(self, idx):
        return self.data[idx]

    def __setitem__(self, idx, value):
        self.data[idx] = value

    def index(self, peak, tolerance=None):

        if tolerance is None:
            tolerance = self.get_tolerance()

        if isinstance(tolerance, float):
            tolerance = np.ones(len(peak)) * tolerance

        # Find index of peak within tolerance limit
        diff = np.abs(self.data - peak) < tolerance
        diff = np.sum(diff, axis=1) == self.dim
        return -1 if len(np.where(diff)[0]) == 0 else np.where(diff)[0][0]

    def get_num_of_assigned_peaks(self):
        if self.has(PeakList.ASSIGNMENT):
            return sum(1 if all(k is not None for k in e) else 0 for e in self.get(PeakList.ASSIGNMENT))
        else:
            return 0

    def get_peak_assignment(self, id):

        if not self.has(PeakList.ASSIGNMENT):
            raise Exception("No assignment is stored in the peak list.")

        assignment = self.get(PeakList.ASSIGNMENT)[id]
        assignment = [e.split(".") if e is not None else None for e in assignment]

        return [(e[0], int(e[1])) if e is not None and len(e) == 2 else None for e in assignment]
