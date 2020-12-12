import pickle


class Utils:
    def __init__(self):
        pass

    def save_state_to_file(self, path, content):
        with open(path, 'wb') as f:
            pickle.dump(content, f)

    def read_state_from_file(self, path):
        with open(path, 'rb') as f:
            content = pickle.load(f)
            return content