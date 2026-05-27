ycounter=2**9+1

with open("./3InARow_training_data", "a") as f:
    for i in range(0, 2**9):
        o=bin(i)[2:].zfill(9)
        o+=";"
        if "111" in (o[0:3], o[3:6], o[6:9]):
            o+="Y"
        elif o[0]+o[3]+o[6] == "111" or o[1]+o[4]+o[7] == "111" or o[2]+o[5]+o[8] == "111":
            o+="Y"
        elif o[0]+o[4]+o[8] == "111" or o[2]+o[4]+o[6] == "111":
            o+="Y"
        else:
            o+="N"
            ycounter-=1
        o+="\n"
        #f.write(o)
