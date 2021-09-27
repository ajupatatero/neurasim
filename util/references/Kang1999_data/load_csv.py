import csv
import numpy as np
import matplotlib.pyplot as plt

import pdb

# with open('Phase_diagram_re_100.csv', newline='') as csvfile:
#     Cd_A_0_0_RE_100 = []
#     Cl_A_0_0_RE_100 = []
#     Cd_A_0_5_RE_100 = []
#     Cl_A_0_5_RE_100 = []
#     Cd_A_1_0_RE_100 = []
#     Cl_A_1_0_RE_100 = []
#     Cd_A_1_5_RE_100 = []
#     Cl_A_1_5_RE_100 = []
            
#     reader = csv.reader(csvfile, delimiter=',')
#     for i, row in enumerate(reader):
#         if i <=1:
#             continue
#         else:
#             #print(f'{row}')
#             # RE_100_A_0.0,,  RE_100_A_0.5,,  RE_100_A_1.0,,  RE_100_A_1.5,
#             # X,Y,            X,Y,            X,Y,            X,Y
#             #X==Cd Y==Cl

#             Cd_A_0_0_RE_100.append(row[0])
#             Cl_A_0_0_RE_100.append(row[1])
#             Cd_A_0_5_RE_100.append(row[2])
#             Cl_A_0_5_RE_100.append(row[3])
#             Cd_A_1_0_RE_100.append(row[4])
#             Cl_A_1_0_RE_100.append(row[5])
#             Cd_A_1_5_RE_100.append(row[6])
#             Cl_A_1_5_RE_100.append(row[7])

#     #Remove "" positions
#     Cd_A_0_0_RE_100 = [float(x) for x in Cd_A_0_0_RE_100 if x]
#     Cl_A_0_0_RE_100 = [float(x) for x in Cl_A_0_0_RE_100 if x]
#     Cd_A_0_5_RE_100 = [float(x) for x in Cd_A_0_5_RE_100 if x]
#     Cl_A_0_5_RE_100 = [float(x) for x in Cl_A_0_5_RE_100 if x]
#     Cd_A_1_0_RE_100 = [float(x) for x in Cd_A_1_0_RE_100 if x]
#     Cl_A_1_0_RE_100 = [float(x) for x in Cl_A_1_0_RE_100 if x]
#     Cd_A_1_5_RE_100 = [float(x) for x in Cd_A_1_5_RE_100 if x]
#     Cl_A_1_5_RE_100 = [float(x) for x in Cl_A_1_5_RE_100 if x]



#     np.save(f'./reference/A_0.0_RE_100.0_Cd_Kang1999.npy', Cd_A_0_0_RE_100)
#     np.save(f'./reference/A_0.0_RE_100.0_Cl_Kang1999.npy', Cl_A_0_0_RE_100)
#     np.save(f'./reference/A_0.5_RE_100.0_Cd_Kang1999.npy', Cd_A_0_5_RE_100)
#     np.save(f'./reference/A_0.5_RE_100.0_Cl_Kang1999.npy', Cl_A_0_5_RE_100)
#     np.save(f'./reference/A_1.0_RE_100.0_Cd_Kang1999.npy', Cd_A_1_0_RE_100)
#     np.save(f'./reference/A_1.0_RE_100.0_Cl_Kang1999.npy', Cl_A_1_0_RE_100)
#     np.save(f'./reference/A_1.5_RE_100.0_Cd_Kang1999.npy', Cd_A_1_5_RE_100)
#     np.save(f'./reference/A_1.5_RE_100.0_Cl_Kang1999.npy', Cl_A_1_5_RE_100)


#     #Check import
#     fig, ax = plt.subplots()

#     for alpha_i in [0.0, 0.5, 1.0, 1.5]:
#         Cl_kang = np.load(f'./reference/A_{alpha_i}_RE_100.0_Cl_Kang1999.npy')
#         Cd_kang = np.load(f'./reference/A_{alpha_i}_RE_100.0_Cd_Kang1999.npy')

#         ax.plot(Cd_kang, Cl_kang, 'k--', label=r'$\alpha$'+f' = {alpha_i} (Kang 1999)')

#     ax.set(xlabel='Cd [·]', ylabel='Cl [·]', title=f'Phase Diagram [ Re=100 ]')
#     ax.legend()
#     fig.savefig(f"./reference/RE_100.0_CHECK_PhaseDiagram.png")
#     plt.close()




RE=[100, 160] #200
A=[0.0, 0.5, 1.0, 1.5]

for j, Re in enumerate(RE):
    with open(f'Phase_diagram_re_{Re}.csv', newline='') as csvfile:
        Cd = []
        Cl = []
        for a, _ in enumerate(A):
            Cd.append([])
            Cl.append([])

        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i <=1:
                continue
            else:
                #print(f'{row}')
                # RE_100_A_0.0,,  RE_100_A_0.5,,  RE_100_A_1.0,,  RE_100_A_1.5,
                # X,Y,            X,Y,            X,Y,            X,Y
                #X==Cd Y==Cl

                for a, _ in enumerate(A):
                    Cd[a].append(row[2*a])
                    Cl[a].append(row[2*a+1])

        #Remove "" positions
        for a, _ in enumerate(A):
            Cd[a] = [float(x) for x in Cd[a] if x]
            Cl[a] = [float(x) for x in Cl[a] if x]

        for a, alpha in enumerate(A):
            np.save(f'./reference/A_{alpha}_RE_{Re}.0_Cd_Kang1999.npy', Cd[a])
            np.save(f'./reference/A_{alpha}_RE_{Re}.0_Cl_Kang1999.npy', Cl[a])



        #Check import
        fig, ax = plt.subplots()

        for alpha_i in A:
            Cl_kang = np.load(f'./reference/A_{alpha_i}_RE_{Re}.0_Cl_Kang1999.npy')
            Cd_kang = np.load(f'./reference/A_{alpha_i}_RE_{Re}.0_Cd_Kang1999.npy')

            ax.plot(Cd_kang, Cl_kang, 'k--', label=r'$\alpha$'+f' = {alpha_i} (Kang 1999)')

        ax.set(xlabel='Cd [·]', ylabel='Cl [·]', title=f'Phase Diagram [ Re={Re} ]')
        ax.legend()
        fig.savefig(f"./reference/RE_{Re}.0_CHECK_PhaseDiagram.png")
        plt.close()
