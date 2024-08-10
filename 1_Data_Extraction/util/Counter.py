class Counter():
    count_n = 0

    @staticmethod
    def count():
        Counter.count_n += 1
        return Counter.count_n