from CifFile import ReadCif

class Cif:
    def __init__(self, file_path=None):
        """
        Initializes a new instance of the Cif class, which reads and processes data from a CIF file.
        
        :param file_path: The file path to the CIF file.
        :raises ValueError: If no file path is provided or the file is not properly formatted.
        :raises IOError: If the file cannot be read.
        """
        if not file_path:
            raise ValueError("A file path must be provided to read CIF data.")
        
        try:
            cif = ReadCif(file_path)
            if not cif:
                raise ValueError("CIF data is empty or incorrectly formatted.")
            
            # Assumes the first block is the relevant one
            data_block = next(iter(cif.keys()), None)
            if data_block is None:
                raise ValueError("No data blocks found in the CIF file.")
            
            self.data = cif[data_block]
        
            self.a = self._parse_float("_cell_length_a")
            self.b = self._parse_float("_cell_length_b")
            self.c = self._parse_float("_cell_length_c")
            self.alpha = self._parse_float("_cell_angle_alpha")
            self.beta = self._parse_float("_cell_angle_beta")
            self.gamma = self._parse_float("_cell_angle_gamma")
            self.space_group = self._parse_int("_space_group_IT_number")
        
        except Exception as e:
            raise IOError(f"Failed to read or parse the CIF file: {str(e)}")

    def _parse_float(self, key):
        """
        Safely parses a float from the CIF data by cleaning the string.

        :param key: The key in the CIF data.
        :return: The parsed float value.
        """
        try:
            return float(self.data.get(key, "0").split("(")[0])
        except ValueError:
            raise ValueError(f"Error parsing float from key {key}")

    def _parse_int(self, key):
        """
        Safely parses an integer from the CIF data.

        :param key: The key in the CIF data.
        :return: The parsed integer value.
        """
        try:
            return int(self._parse_float(key))
        except ValueError:
            raise ValueError(f"Error parsing int from key {key}")



from CifFile import ReadCif

class Cif:
    def __init__(self, file_path=None):
        """
        Initializes a new instance of the Cif class, which reads and processes data from a CIF file.
        
        :param file_path: The file path to the CIF file.
        :raises ValueError: If no file path is provided or the file is not properly formatted.
        :raises IOError: If the file cannot be read.
        """
        if not file_path:
            raise ValueError("A file path must be provided to read CIF data.")
        
        try:
            cif = ReadCif(file_path)
            if not cif:
                raise ValueError("CIF data is empty or incorrectly formatted.")
            
            data_blocks = list(cif.keys())
            if not data_blocks:
                raise ValueError("No data blocks found in the CIF file.")
            
            # Attempt to find a valid data block
            self.data = None
            for idx, block in enumerate(data_blocks):
                try:
                    current_data = cif[block]
                    a = self._parse_float(current_data, "_cell_length_a")
                    b = self._parse_float(current_data, "_cell_length_b")
                    c = self._parse_float(current_data, "_cell_length_c")
                    alpha = self._parse_float(current_data, "_cell_angle_alpha")
                    beta = self._parse_float(current_data, "_cell_angle_beta")
                    gamma = self._parse_float(current_data, "_cell_angle_gamma")
                    
                    # Check if any lattice parameter is zero
                    if any(param == 0.0 for param in [a, b, c, alpha, beta, gamma]):
                        if idx + 1 < len(data_blocks):
                            continue  # Try the next block
                        else:
                            raise ValueError(
                                f"Lattice parameters contain zero in block '{block}' "
                                f"and no subsequent data blocks are available."
                            )
                    else:
                        # Valid data block found
                        self.data = current_data
                        self.a = a
                        self.b = b
                        self.c = c
                        self.alpha = alpha
                        self.beta = beta
                        self.gamma = gamma
                        self.space_group = self._parse_int(current_data, "_space_group_IT_number")
                        break
                except ValueError as ve:
                    # If parsing fails, try the next block
                    if idx + 1 < len(data_blocks):
                        continue
                    else:
                        raise ValueError(
                            f"Error parsing lattice parameters in block '{block}': {ve}"
                        )
            
            if self.data is None:
                raise ValueError("No valid data block with non-zero lattice parameters found.")
        
        except Exception as e:
            raise IOError(f"Failed to read or parse the CIF file: {str(e)}")
    
    def _parse_float(self, data, key):
        """
        Safely parses a float from the CIF data by cleaning the string.

        :param data: The data block from the CIF file.
        :param key: The key in the CIF data.
        :return: The parsed float value.
        :raises ValueError: If the key is missing or the value cannot be parsed as float.
        """
        try:
            value_str = data.get(key, "0").split("(")[0]
            value = float(value_str)
            return value
        except ValueError:
            raise ValueError(f"Error parsing float from key '{key}' with value '{data.get(key, '0')}'")
    
    def _parse_int(self, data, key):
        """
        Safely parses an integer from the CIF data.

        :param data: The data block from the CIF file.
        :param key: The key in the CIF data.
        :return: The parsed integer value.
        :raises ValueError: If the key is missing or the value cannot be parsed as int.
        """
        try:
            return int(self._parse_float(data, key))
        except ValueError:
            raise ValueError(f"Error parsing int from key '{key}' with value '{data.get(key, '0')}'")
