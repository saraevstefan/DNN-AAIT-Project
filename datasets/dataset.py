class Dataset:
    def __init__(self, data_dir):
        """
        Base class for dataset handling.
        
        :param data_dir: Path to the directory containing the dataset files.
        """
        self.data_dir = data_dir
        self.data = {}
        self._load_data()
        self._preprocess()

    def _load_data(self):
        """
        Load the dataset. This method should be implemented by subclasses.
        """
        raise NotImplementedError("load_data() must be implemented by subclasses.")

    def _preprocess(self):
        """
        Preprocess the dataset. This can include tokenization, normalization, etc.
        """
        raise NotImplementedError("preprocess() must be implemented by subclasses.")

    def get_data(self):
        """
        Return the data for training/testing.
        """
        raise NotImplementedError("get_data() must be implemented by subclasses.")
