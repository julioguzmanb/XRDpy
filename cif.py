from CifFile import ReadCif
import numpy as np

# Module-level constants for required keys
REQUIRED_LATTICE_KEYS = [
    "_cell_length_a", "_cell_length_b", "_cell_length_c",
    "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma"
]

REQUIRED_ATOM_KEYS = [
    "_atom_site_label",
    "_atom_site_fract_x",
    "_atom_site_fract_y",
    "_atom_site_fract_z"
]

def read_cif_file(file_path):
    """
    Reads and validates the CIF file.

    :param file_path: Path to the CIF file.
    :return: Parsed CIF data.
    :raises IOError: If the file cannot be read or is improperly formatted.
    """
    try:
        cif = ReadCif(file_path)
        if not cif:
            raise ValueError("CIF data is empty or incorrectly formatted.")
        return cif
    except Exception as e:
        raise IOError(f"Failed to read CIF file '{file_path}': {e}") from e

def find_valid_data_block(cif_data):
    """
    Finds the first valid data block with non-zero lattice parameters.

    :param cif_data: Parsed CIF data.
    :return: Valid data block dictionary.
    :raises ValueError: If no valid data block is found.
    """
    for block_name, block_data in cif_data.items():
        try:
            lattice_params = [
                parse_value(block_data, key, float)
                for key in REQUIRED_LATTICE_KEYS
            ]
            if all(param != 0.0 for param in lattice_params):
                return block_data
        except ValueError:
            continue  # Invalid block, proceed to the next one
    raise ValueError("No valid data block with non-zero lattice parameters found.")

def extract_lattice_parameters(data_block):
    """
    Extracts lattice parameters from the data block.

    :param data_block: Valid data block dictionary.
    :return: Tuple of lattice parameters (a, b, c, alpha, beta, gamma).
    """
    try:
        lattice_params = [
            parse_value(data_block, key, float)
            for key in REQUIRED_LATTICE_KEYS
        ]
        return tuple(lattice_params)
    except ValueError as ve:
        raise ValueError(f"Error extracting lattice parameters: {ve}") from ve

def extract_atomic_positions(data_block):
    """
    Extracts atomic positions from the data block.

    :param data_block: Valid data block dictionary.
    :return: Dictionary mapping atom labels to their fractional coordinates as NumPy arrays.
    :raises ValueError: If required atom keys are missing or data lengths mismatch.
    """
    # Ensure all required atom keys are present
    missing_keys = [key for key in REQUIRED_ATOM_KEYS if key not in data_block]
    if missing_keys:
        raise ValueError(f"Missing required atom keys: {', '.join(missing_keys)}")

    labels = data_block["_atom_site_label"]
    fract_x = data_block["_atom_site_fract_x"]
    fract_y = data_block["_atom_site_fract_y"]
    fract_z = data_block["_atom_site_fract_z"]

    # Validate lengths
    if not (len(labels) == len(fract_x) == len(fract_y) == len(fract_z)):
        raise ValueError("Atom site lists have mismatched lengths.")

    atom_positions = {}
    for label, x_str, y_str, z_str in zip(labels, fract_x, fract_y, fract_z):
        try:
            x = parse_value(x_str, value_type=float, is_fraction=True)
            y = parse_value(y_str, value_type=float, is_fraction=True)
            z = parse_value(z_str, value_type=float, is_fraction=True)
            atom_positions[label] = np.array([x, y, z])
        except ValueError as ve:
            raise ValueError(f"Error parsing atomic position for atom '{label}': {ve}") from ve

    return atom_positions

def parse_value(source, key=None, value_type=float, is_fraction=False):
    """
    Parses a value from the CIF data or a standalone string, converting it to the specified type.

    :param source: The data dictionary from the CIF file or a standalone string.
    :param key: The key in the CIF data (optional if source is a string).
    :param value_type: The type to convert the value to (e.g., float, int).
    :param is_fraction: Indicates if the value is a fractional coordinate.
    :return: The parsed value in the specified type.
    :raises ValueError: If the key is missing or the value cannot be parsed.
    """
    try:
        if key:
            # Parsing from the data dictionary
            value_str = source.get(key, "0").split("(")[0].strip()
        else:
            # Parsing from a standalone string
            value_str = source.split("(")[0].strip()

        parsed_value = value_type(value_str)

        # If parsing fractional coordinates, validate their range
        if is_fraction:
            if not (0.0 <= parsed_value <= 1.0):
                raise ValueError(f"Fractional coordinate has an out-of-range value: {parsed_value}")

        return parsed_value
    except ValueError as ve:
        if key:
            raise ValueError(
                f"Error parsing {value_type.__name__} from key '{key}' with value '{source.get(key, '0')}'"
            ) from ve
        else:
            raise ValueError(f"Error parsing {value_type.__name__} from value '{source}'") from ve

class Cif:
    """
    A class to represent and process CIF (Crystallographic Information File) data.
    """

    def __init__(self, file_path):
        """
        Initializes a new instance of the Cif class, which reads and processes data from a CIF file.

        :param file_path: The file path to the CIF file.
        :raises ValueError: If the file is not properly formatted.
        :raises IOError: If the file cannot be read.
        """
        if not file_path:
            raise ValueError("A file path must be provided to read CIF data.")

        # Read and validate CIF file
        cif_data = read_cif_file(file_path)

        # Find a valid data block
        try:
            self.data = find_valid_data_block(cif_data)
        except ValueError as ve:
            raise ValueError(ve) from ve

        # Extract lattice parameters
        try:
            self.a, self.b, self.c, self.alpha, self.beta, self.gamma = extract_lattice_parameters(self.data)
        except ValueError as ve:
            raise ValueError(ve) from ve

        # Extract space group
        try:
            self.space_group = parse_value(self.data, "_space_group_IT_number", int)
        except ValueError as ve:
            raise ValueError(f"Error parsing space group: {ve}") from ve

        # Extract atomic positions
        try:
            self.atom_positions = extract_atomic_positions(self.data)
        except ValueError as ve:
            raise ValueError(ve) from ve

    def get_lattice_parameters(self):
        """
        Returns the lattice parameters.

        :return: Tuple of lattice parameters (a, b, c, alpha, beta, gamma).
        """
        return self.a, self.b, self.c, self.alpha, self.beta, self.gamma

    def get_space_group(self):
        """
        Returns the space group number.

        :return: Space group IT number as an integer.
        """
        return self.space_group

    def get_atom_positions(self):
        """
        Returns the atomic positions.

        :return: Dictionary mapping atom labels to their fractional coordinates as NumPy arrays.
        """
        return self.atom_positions
