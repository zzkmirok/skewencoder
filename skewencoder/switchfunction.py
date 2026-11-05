import numpy as np

__all__ = ["SwitchFun"]

class SwitchFun:
    def __init__(self, r0, FunType = "rational", options: dict = None):
        self.r0 = r0
        self.m = 12
        self.n = 6
        self.d0 = 0.
        if FunType == "rational":
            self.FunType = "rational"
            if options is not None:
                for o in options.keys():
                    if o == "m":
                        self.m = options[o]
                    if o == "n":
                        self.n = options[o]
                    if o == "d0":
                        self.d0 = options[o]
        else:
            raise ValueError()
    
    def set_r0(self, r0: float):
        self.r0 = r0
    
    def get_r0(self):
        return self.r0

    def __str__(self):
        print_str = "Switch function:\n"
        print_str += f"r0 = {self.r0}, d0 = {self.d0}, m,n = {(self.m, self.n)}\n"
        print_str += f"funtype: {self.FunType}\n"
        return print_str

    def __call__(self, distances_lists : np.array = None):
        """PLUMED RATIONAL: s = (1 - x^n) / (1 - x^m) with x = (r - d0) / r0.

        As in PLUMED, s = 1 for r <= d0; otherwise even powers of a negative x would make s fall
        again for r < d0. At x = 1 the 0/0 is replaced by its limit n/m.
        """
        if distances_lists is not None:
            if self.FunType == "rational":
                shifted = np.maximum((np.asarray(distances_lists, dtype=float) - self.d0) / self.r0, 0.)
                num = 1. - shifted ** self.n
                den = 1. - shifted ** self.m
                at_one = np.isclose(den, 0.)
                s = np.where(at_one, self.n / self.m, num / np.where(at_one, 1., den))
                return s[()]
            else:
                raise ValueError
    
            
def test_SwitchFun():
    x = np.array([[1.1, 1.8], [1.2, 2.0]])
    adjlists = SwitchFun(r0 = 1.7, options= {"m": 8, "n": 6})(x)
    print(SwitchFun(r0 = 1.7, options= {"m": 8, "n": 6}))
    print(adjlists)
    adjlists1 = SwitchFun(r0 = 1.7)(x)
    print(SwitchFun(r0 = 1.7))
    print(adjlists1)

if __name__ == "__main__":
    test_SwitchFun()
    # a = np.arange(0,6).reshape((3,2))
    # print(a[-2:, :])