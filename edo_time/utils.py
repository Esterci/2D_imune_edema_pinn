class ProgBar:
    def __init__(self, n_elements, int_str):
        import sys

        self.n_elements = n_elements
        self.progress = 0

        print(int_str)

        # initiallizing progress bar

        info = "{:.2f}% - {:d} of {:d}".format(0, 0, n_elements)

        formated_bar = " " * int(50)

        sys.stdout.write("\r")

        sys.stdout.write("[%s] %s" % (formated_bar, info))

        sys.stdout.flush()

    def update(self, prog_info=None):
        import sys

        if prog_info == None:
            self.progress += 1

            percent = (self.progress) / self.n_elements * 100 / 2

            info = "{:.2f}% - {:d} of {:d}".format(
                percent * 2, self.progress, self.n_elements
            )

            formated_bar = "-" * int(percent) + " " * int(50 - percent)

            sys.stdout.write("\r")

            sys.stdout.write("[%s] %s" % (formated_bar, info))

            sys.stdout.flush()

        else:
            self.progress += 1

            percent = (self.progress) / self.n_elements * 100 / 2

            info = (
                "{:.2f}% - {:d} of {:d} ".format(
                    percent * 2, self.progress, self.n_elements
                )
                + prog_info
            )

            formated_bar = "-" * int(percent) + " " * int(50 - percent)

            sys.stdout.write("\r")

            sys.stdout.write("[%s] %s" % (formated_bar, info))

            sys.stdout.flush()
