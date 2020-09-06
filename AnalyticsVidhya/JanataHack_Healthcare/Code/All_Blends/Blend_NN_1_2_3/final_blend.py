import numpy as np

class_map_rev = {0: "0-10", 1: "11-20", 2: "21-30", 3: "31-40", 4: "41-50", 5: "51-60", 6: "61-70", 7: "71-80", 8: "81-90", 9: "91-100", 10: "More than 100 Days"}

fp = open("submit.csv", "w")
fp.write("case_id,Stay\n")
fp_probs = open("submit_proba.csv", "w")
fp_probs.write("case_id,0-10,11-20,21-30,31-40,41-50,51-60,61-70,71-80,81-90,91-100,More Than 100 Days\n")


fp1 = open("final_submit_proba_lower.csv")
fp1.readline()

fp2 = open("final_submit_proba_higher.csv")
fp2.readline()

fp3 = open("final_submit_probaV3.csv")
fp3.readline()

while True:
    line1, line2, line3 = fp1.readline(), fp2.readline(), fp3.readline()
    if line1 and line2 and line3:
        line1 = line1.strip().split(',')
        line2 = line2.strip().split(',')
        line3 = line3.strip().split(',')
        a, b = np.array([float(i) for i in line1[1:]]), np.array([float(i) for i in line2[1:]])
        c = a*0.5 + b*0.5
        d = np.array([float(i) for i in line3[1:]])
        e = c*0.5 + d*0.5
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
