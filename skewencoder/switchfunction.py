import numpy as np

__all__ = ["SwitchFun"]

class SwitchFun:
    def __init__(self, r0, FunType = "rational", options: dict = None):
        self.r0 = r0
        self.m = 12
        self.n = 6
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
        print_str += f"r0 = {self.r0}, m,n = {(self.m, self.n)}\n"
        print_str += f"funtype: {self.FunType}\n"
        return print_str

    def __call__(self, distances_lists : np.array = None):
        if distances_lists is not None:
            if self.FunType == "rational":
                return (1. - (distances_lists / self.r0) ** self.n) / (1. - (distances_lists / self.r0) ** self.m)
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