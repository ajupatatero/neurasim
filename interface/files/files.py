import numpy as np

def export_csv(file_path,field, Lx, Ly, dx, dy):
    timestep=0


    #Create position vectors and arrays
    x = np.arange(dx/2, Lx-dx/2 +dx/2, dx)
    y = np.arange(dy/2, Ly-dy/2 +dy/2, dy)
    X, Y = np.meshgrid(x, y)

    Z = field.values._native.cpu().numpy()[0]
    Z = np.rot90(Z)

    CSV = np.zeros((Z.shape[0]*Z.shape[1],3))

    count=0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            CSV[count][:] = [X[i][j], Y[i][j], Z[i][j]]
            count = count + 1

    np.savetxt(f"test.csv.{timestep}", CSV, delimiter=",", header='x coord,y coord,scalar') #, comments='')
