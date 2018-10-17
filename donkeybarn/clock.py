class FuncTimer():
    def __init__(self):
        self._results = []
        self.record = {}

    @property
    def elapsed(self):
        return time.time() - self.start

    def start(self, name=''):
        self.record[name] = {}
        self.record[name]['start'] = time.time()
        self.record[name]['name'] = name

    def stop(self, name=''):
        self.record[name]['finish'] = time.time()
        duration = self.record[name]['finish'] - self.record[name]['start']
        self.record[name]['duration'] = duration
        self._results.append(self.record[name])
        self.record = {}

    @property
    def results(self):
        return pd.DataFrame(self._results)

    def save_results(self, file_path=None):
        df = self.results
        df.to_csv(file_path)