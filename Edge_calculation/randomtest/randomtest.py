import random
import numpy as np

class RandomReduction:

    def reduction(self,num, reductpercentage, width, inputfile, outputfile):

        if reductpercentage < 0 or reductpercentage > 1:
            print("Reductpercentage should be greater than 0 and less than 1.Please input the reductpercentage again:")
            return 0
        # head = "d:dataS1-"
        # shu = str(num)
        # wei = ".csv"
        frw = open(outputfile, 'w')

        try:
            fr = open(inputfile, 'r')
            line = fr.readlines()

        except FileNotFoundError:
            msg = "Sorry,the inputfile "+inputfile+"does not exist."
            print(msg)
            return 0

        try:
            fry = open(tagfile, 'r')
            liney = fry.readlines()
        except FileNotFoundError:
            msg = "Sorry,the tagfile "+tagfile+"does not exist."
            print(msg)
            return 0

        if width > len(line):
            print("The width of windows should be less than total data.Please input the width again:")
            return 0

        #根据属性值个数设置数组个数
        z1 = [0.0 for i in range(len(line))]
        z2 = [0.0 for i in range(len(line))]
        z3 = [0.0 for i in range(len(line))]
        z4 = [0.0 for i in range(len(line))]
        z5 = [0.0 for i in range(len(line))]
        z6 = [0.0 for i in range(len(line))]
        y = [0.0 for i in range(len(liney))]
        z = [0.0 for i in range(len(line))]
        print("The program is performing random reduction......")

        # 读取对应的属性值
        # i = 0
        # for L in liney:
        #     string = L.strip("\n")
        #     y[i] = np.float64(string)
        #     # print(y[i])
        #     i = i + 1

        i = 0
        for L in line:
            string = L.strip("\n").split(",")
            z1[i] = np.float64(string[0])
            z2[i] = np.float64(string[1])
            z3[i] = np.float64(string[2])
            z4[i] = np.float64(string[3])
            z5[i] = np.float64(string[4])
            z6[i] = np.float64(string[5])
            # z[i] = z1[i] + z2[i]
            i = i + 1

        maxNum = int((len(line)/width))*width
        for i in range(0, len(line), width):
            print(i)
            if maxNum == i:
                print("******************************************************")
                width = len(line)-int(maxNum)
                print(width)
            flag = [0 for j in range(width)]
            new = [0 for j in range(width)]
            # print(new)
            k = 0
            # print(i+width)
            for j in range(i, i+width):
                new[k] = z[j]
                k = k+1
            number = 0
            # print(new)
            while number < width*reductpercentage:
                random_number = random.randint(0, width-1)
                if flag[random_number] == 0:
                    flag[random_number] = 1
                    number = number + 1
            if number >= width*reductpercentage:
                for j in range(width):
                    if flag[j] == 0:
                        str1 = '%f,%f,%f,%f,%f,%f\n' % (z1[j+i], z2[j+i], z3[j+i], z4[j+i], z5[j+i], z6[j+i])
                        frw.write(str1)
        print("The data reduction is finished.")

r=RandomReduction()
# for i in range(90, 100):
r.reduction(99999, 0.95, 100, "d:trans.csv", "d:dataS2_5896.csv")
