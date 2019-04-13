# Data-driven and Model-based methods to select optimal sensor positions are compared in [Sun & Martius](https://ieeexplore.ieee.org/abstract/document/8625064)
  
  <p align="center"><img src="../Pics/Positioning_methods.png" width="800" height="400" align="center">
  
- [Data](../Data/)

- Code:
  - Greedy Support Vector Regression
  - PCA QR Pivoting(Choose first 30 optimal positions)
  ```Jupyter Notebook
  # Data: deformation_data [3255 forces, 2162 sensor nodals], sensor_position [2162 nodals, 4 dimensions]
  # Package: from sklearn.decomposition import PCA, import numpy
  
  # Function: Select and save 1-30 optimal sensor positions
  for i in range(30):
    sklearn_pca = PCA(n_components=i+1)
    sklearn_pca.fit(deformation_data)
    yy = sklearn_pca.transform(np.eye(deformation_data.shape[1]))
    Q,R,P = sp.linalg.qr(yy.T, pivoting=True)
    position_indicis = P[0:i+1]
    sensor_nr_counter = 0
    posistion_selected = np.zeros((i+1,4))
    for k in position_indicis:
        posistion_selected[sensor_nr_counter,:] = sensor_position[int(k),:]
        sensor_nr_count += 1
    np.savetxt("PCA_Positions/"+str(i+1)+".txt",posistion_selected,delimiter=' ')
    
  # Function: Report reconstruction error w.r.t. optimal selected sensor nr.
  error = np.zeros((500))
  for i in range(1,500):
    sklearn_pca = PCA(n_components=i)
    sklearn_pca.fit(deformation_data)
    yy = sklearn_pca.transform(np.eye(deformation_data.shape[1]))
    Q,R,P = sp.linalg.qr(yy.T, pivoting=True)
    P = P[:,None].T
    c = np.zeros((i,deformation_data.shape[1]))
    for j in range(i):
        c[j,P[0,j]] = 1
    s1 = np.dot(c,deformation_data.T)
    s2 = np.linalg.inv(np.dot(c,yy))
    s3 = np.dot(s2 ,s1)
    x_pre = np.dot(yy, s3)
    error[i] = sum(np.linalg.norm(np.subtract(x_pre.T,deformation_data),axis = 1))/sum(np.linalg.norm(deformation_data,axis = 1))
  ```
  - Entropy Minimization
  - Mutual information Maximization

With no hesitation to contact **huanbo.sunrwth@gmail.com** for unclear explaination. 
