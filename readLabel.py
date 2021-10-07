import copy
#give picture ID, return corresponding label
class GetLabel:
    def __init__(self, f_path = "label_2/"):
        self.label_path = f_path

    #parse txt file
     #type of ids: 1. list, 2.int
    def parse(self, ids):
        file_path_ls, label_ls = [], []
        if type(ids) == int:
            ids = [ids]
        elif type(ids) == list:
            pass
        else:
            #error
            return None
        for i in ids:
            i_str = str(i)
            while len(i_str) < 6:
                i_str = "0" + i_str
            file_path_ls.append(self.label_path + i_str + ".txt")
        for fl in file_path_ls:
            f = open(fl, 'r')
            for l in f.readlines():
                try:
                    c = l.split("\n")[0].split(" ")
                except:
                    #error about the format of txt file
                    c = ["error", "-1", "-1", "-1", "-1", "-1", "-1", "-1", "-1", "-1", "-1", "-1", "-1", "-1", "-1"]
                label_ls.append(c)
            f.close()
        return label_ls

if __name__ == "__main__":
    gs = GetLabel()
    print(gs.parse(1))

        
        