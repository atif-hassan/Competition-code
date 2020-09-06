import numpy as np

class_map_rev = {0: "0-10", 1: "11-20", 2: "21-30", 3: "31-40", 4: "41-50", 5: "51-60", 6: "61-70", 7: "71-80", 8: "81-90", 9: "91-100", 10: "More than 100 Days"}

fp = open("submit.csv", "w")
fp_probs = open("submit_proba.csv", "w")
fp_probs.write("case_id,0-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100,More Than 100 Days\n")


fp1 = open("submit_proba_3.csv")
fp.write(fp1.readline())

fp2 = open("submit_proba_6.csv")
fp2.readline()

fp3 = open("submit_proba_8.csv")
fp3.readline()

while True:
    line1, line2, line3 = fp1.readline(), fp2.readline(), fp3.readline()
    if line1 and line2 and line3:
        line1 = line1.strip().split(',')
        line2 = line2.strip().split(',')
        line3 = line3.strip().split(',')
        if line1[0] == "318446" or line1[0] == "318447" or line1[0] == "318448" or line1[0] == "318449" or line1[0] == "318450" or line1[0] == "318451" or line1[0] == '318452':
            fp.write(line1[0]+",0-10\n")
            fp_probs.write(line1[0]+","+",".join(['0' for i in range(11)])+"\n")
        else:
            a, b = np.array([float(i) for i in line1[1:]]), np.array([float(i) for i in line2[1:]])
            c = a*0.6 + b*0.4
            d = np.array([float(i) for i in line3[1:]])
            e = c*0.6 + d*0.4
            f = np.argmax(e)
            fp.write(line1[0]+","+class_map_rev[f]+"\n")
            fp_probs.write(line1[0]+","+",".join([str(i) for i in e])+"\n")
    else:
        fp1.close()
        fp2.close()
        fp3.close()
        fp.close()
        fp_probs.close()
        break
