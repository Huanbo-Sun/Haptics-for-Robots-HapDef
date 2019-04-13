# Data-driven and Model-based methods to select optimal sensor positions are compared in [Sun & Martius](https://ieeexplore.ieee.org/abstract/document/8625064)
  
  <p align="center"><img src="../Pics/Positioning_methods.png" width="800" height="400" align="center">
  
- [Data](../Data/)

- Code:
  - Greedy Support Vector Regression
  ```Jupyter Notebook
  # Data: deformation_data [3255 forces, 2162 sensor nodals], sensor_position [2162 nodals, 4 dimensions], force_data [3255 forces, 4 dimensions]
  # Package: from sklearn.model_selection import train_test_split, from sklearn.svm import SVR, from sklearn.multioutput import MultiOutputRegressor, import numpy
  
  # Function: Machine learning parameter confirmation using k-fold validation
  x_train1 = deformation_data
  y_train1 = force_data[;,1:]
  x_train2,x_valid1,y_train2,y_valid1 = train_test_split(x_train1,y_train1,test_size=0.2)
  x_train3,x_valid2,y_train3,y_valid2 = train_test_split(x_train2,y_train2,test_size=0.25)
  x_train4,x_valid3,y_train4,y_valid3 = train_test_split(x_train3,y_train3,test_size=0.333)
  x_valid5,x_valid4,y_valid5,y_valid4 = train_test_split(x_train4,y_train4,test_size=0.5)
  
  x_train0 =np.vstack((x_valid2,x_valid3,x_valid4,x_valid5))
  y_train0 =np.vstack((y_valid2,y_valid3,y_valid4,y_valid5))
  l = []
  for idc  in range(5):
      error = []
      for ide in range(8):
          print(ide)
          clf = SVR(C=10**(idc-2), epsilon =10**(-ide))
          clf.fit(x_train0,y_train0[:,0])
          y_pre0 = clf.predict(x_valid1)
          clf.fit(x_train0,y_train0[:,1])
          y_pre1 = clf.predict(x_valid1)
          clf.fit(x_train0,y_train0[:,2])
          y_pre2 = clf.predict(x_valid1)
          y_pre = np.vstack([y_pre0,y_pre1,y_pre2]).T
          error = np.append(error,np.sum(np.linalg.norm(y_pre - y_valid1, axis=1))/y_valid1.shape[0])
      l = np.append(l,error)
    
  # Function: Select and save 1-30 optimal sensor positions.
  x_train1 = np.loadtxt("../02_Data_Processing/00_train_sensor.txt",dtype=float)
  y_train1 = np.loadtxt("../02_Data_Processing/00_train_force.txt",dtype=float,usecols=(1,2,3))
  x_train2,x_valid1,y_train2,y_valid1 = train_test_split(x_train1,y_train1,test_size=0.2)
  x_train3,x_valid2,y_train3,y_valid2 = train_test_split(x_train2,y_train2,test_size=0.25)
  x_train4,x_valid3,y_train4,y_valid3 = train_test_split(x_train3,y_train3,test_size=0.333)
  x_valid5,x_valid4,y_valid5,y_valid4 = train_test_split(x_train4,y_train4,test_size=0.5)

  error = np.zeros((30))
  clf = SVR(C=0.1, epsilon =10**(-4))
  position_indicis = np.linspace(0,2161,2162)

  x_train01 =np.vstack((x_valid2,x_valid3,x_valid4,x_valid5))
  y_train01 =np.vstack((y_valid2,y_valid3,y_valid4,y_valid5))
  x_train02 =np.vstack((x_valid1,x_valid3,x_valid4,x_valid5))
  y_train02 =np.vstack((y_valid1,y_valid3,y_valid4,y_valid5))
  x_train03 =np.vstack((x_valid1,x_valid2,x_valid4,x_valid5))
  y_train03 =np.vstack((y_valid1,y_valid2,y_valid4,y_valid5))
  x_train04 =np.vstack((x_valid1,x_valid2,x_valid3,x_valid5))
  y_train04 =np.vstack((y_valid1,y_valid2,y_valid3,y_valid5))
  x_train05 =np.vstack((x_valid1,x_valid2,x_valid3,x_valid4))
  y_train05 =np.vstack((y_valid1,y_valid2,y_valid3,y_valid4))

  x_train_PCA_SVR01 = np.zeros((x_train01.shape[0],1))
  x_test_PCA_SVR01 = np.zeros((x_valid1.shape[0],1))
  x_train_PCA_SVR02 = np.zeros((x_train02.shape[0],1))
  x_test_PCA_SVR02 = np.zeros((x_valid2.shape[0],1))
  x_train_PCA_SVR03 = np.zeros((x_train03.shape[0],1))
  x_test_PCA_SVR03 = np.zeros((x_valid3.shape[0],1))
  x_train_PCA_SVR04 = np.zeros((x_train04.shape[0],1))
  x_test_PCA_SVR04 = np.zeros((x_valid4.shape[0],1))
  x_train_PCA_SVR05 = np.zeros((x_train05.shape[0],1))
  x_test_PCA_SVR05 = np.zeros((x_valid5.shape[0],1))
  
  def errorr(x_train_PCA_SVR,x_test_PCA_SVR,x_train,y_train,x_valid,y_valid,k):
    x_train_PCA_SVR = np.hstack((x_train_PCA_SVR,x_train[:,int(k)][:,None]))
    x_test_PCA_SVR = np.hstack((x_test_PCA_SVR,x_valid[:,int(k)][:,None]))
    if x_train_PCA_SVR[0,0]== 0:
        x_train_PCA_SVR = np.delete(x_train_PCA_SVR,0,1)
        x_test_PCA_SVR = np.delete(x_test_PCA_SVR,0,1)
    clf.fit(x_train_PCA_SVR,y_train[:,0])
    y_pre0 = clf.predict(x_test_PCA_SVR)
    clf.fit(x_train_PCA_SVR,y_train[:,1])
    y_pre1 = clf.predict(x_test_PCA_SVR)
    clf.fit(x_train_PCA_SVR,y_train[:,2])
    y_pre2 = clf.predict(x_test_PCA_SVR)
    y_pre = np.vstack([y_pre0,y_pre1,y_pre2]).T
    err = sum(np.linalg.norm(np.subtract(y_pre,y_valid),axis = 1))/y_pre.shape[0]
    x_train_PCA_SVR = np.delete(x_train_PCA_SVR,-1,1)
    x_test_PCA_SVR = np.delete(x_test_PCA_SVR,-1,1)
    return(err) 
    
  nr = np.zeros(30)
  for i in range(30):
      err = np.zeros((aa.shape[0]))
      j=0
      for k in aa:
          err1 = errorr(x_train_PCA_SVR01,x_test_PCA_SVR01,x_train01,y_train01,x_valid1,y_valid1,k)
          err2 = errorr(x_train_PCA_SVR02,x_test_PCA_SVR02,x_train02,y_train02,x_valid2,y_valid2,k)
          err3 = errorr(x_train_PCA_SVR03,x_test_PCA_SVR03,x_train03,y_train03,x_valid3,y_valid3,k)
          err4 = errorr(x_train_PCA_SVR04,x_test_PCA_SVR04,x_train04,y_train04,x_valid4,y_valid4,k)
          err5 = errorr(x_train_PCA_SVR05,x_test_PCA_SVR05,x_train05,y_train05,x_valid5,y_valid5,k)
          err[j] = (err1+err2+err3+err4+err5)/5
          j=j+1

      error[i]=np.min(err)
      aaa = np.argmin(err)
      x_train_PCA_SVR01 = np.hstack((x_train_PCA_SVR01,x_train01[:,int(position_indicis[aaa])][:,None]))
      x_test_PCA_SVR01 = np.hstack((x_test_PCA_SVR01,x_valid1[:,int(position_indicis[aaa])][:,None]))
      x_train_PCA_SVR02 = np.hstack((x_train_PCA_SVR02,x_train02[:,int(position_indicis[aaa])][:,None]))
      x_test_PCA_SVR02 = np.hstack((x_test_PCA_SVR02,x_valid2[:,int(position_indicis[aaa])][:,None]))
      x_train_PCA_SVR03 = np.hstack((x_train_PCA_SVR03,x_train03[:,int(position_indicis[aaa])][:,None]))
      x_test_PCA_SVR03 = np.hstack((x_test_PCA_SVR03,x_valid3[:,int(position_indicis[aaa])][:,None]))
      x_train_PCA_SVR04 = np.hstack((x_train_PCA_SVR04,x_train04[:,int(position_indicis[aaa])][:,None]))
      x_test_PCA_SVR04 = np.hstack((x_test_PCA_SVR04,x_valid4[:,int(position_indicis[aaa])][:,None]))
      x_train_PCA_SVR05 = np.hstack((x_train_PCA_SVR05,x_train05[:,int(position_indicis[aaa])][:,None]))
      x_test_PCA_SVR05 = np.hstack((x_test_PCA_SVR05,x_valid5[:,int(position_indicis[aaa])][:,None]))

      nr[i] = position_indicis[aaa]
      position_indicis = np.delete(position_indicis,aaa)
      
    aaaposition = sensor_position
    greedypos = np.zeros((30,4))
    count = 0
    for k in nr:
        print(k)
        greedypos[count,:]=aaaposition[int(k),:]
        count = count+1
    np.savetxt("Greedy_SVR_kfold_Positions.txt",greedypos, delimiter=' ')
  ```
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
    position_indicis = np.zeros((i,deformation_data.shape[1]))
    for j in range(i):
        position_indicis[j,P[0,j]] = 1
    s1 = np.dot(position_indicis,deformation_data.T)
    s2 = np.linalg.inv(np.dot(position_indicis,yy))
    s3 = np.dot(s2 ,s1)
    deformation_prediction = np.dot(yy, s3)
    error[i] = sum(np.linalg.norm(np.subtract(deformation_prediction.T,deformation_data),axis = 1))/sum(np.linalg.norm(deformation_data,axis = 1))
  ```
  - Entropy Minimization
  ```Jupyter Notebook
  # Data: sensor_position_candidate [2162 nodals, 4 dimensions], sensor_position_all [3726 nodals, 4 dimensions]
  # Package: import numpy
  
  # Function: Kernel definition
  def gp_kernel1(sita,x,y):
      if np.sum(x.shape)<4:
          x = x[None,:]
      if np.sum(y.shape)<4:
          y = y[None,:]
      gpcovariance = np.zeros((x.shape[0],y.shape[0]))
      ry = 0.4292
      rxy1 = np.sqrt(x[:,0]**2+x[:,2]**2)
      rxy2 = np.sqrt(y[:,0]**2+y[:,2]**2)
      angle1 = np.arctan2(x[:,2],x[:,0])
      angle2 = np.arctan2(y[:,2],y[:,0])
      for j in range(y.shape[0]):
          for i in range(x.shape[0]):
              dy_ = np.sqrt((x[i,1]-y[j,1])**2+(rxy1[i]-rxy2[j])**2)
              dy__ = 2*ry*np.arcsin((dy_/2)/ry)

              angle = np.abs(angle1[i] - angle2[j])
              if angle > np.pi:
                  angle = 2*np.pi-angle
              dxz__ = (rxy1[i]+rxy2[j])/2*angle

              index = np.sqrt((dy__**2+dxz__**2)) 
              gpcovariance[i,j] = np.exp(-index**2/sita**2)
       return(np.around(gpcovariance,decimals=10))
    
  # Function: Optimal sensor position selection
  A = np.zeros((1,4))
  uncertainty = []
  sita_min=0.0044 # because of cauculation precision
  sita_max =0.022 # because of entropy bigger than 0
  sita = 0.0175
  k_base = gp_kernel1(sita,S[:,1:4],S[:,1:4])

  firsta = np.argmin(np.sum(k_base,axis=0))
  A = S[firsta,:][None,:]

  for i in range(S.shape[0]):
      if S[i,0] == A[0,0]:
          S =np.delete(S,i,axis=0)
          break

  for j in range(29):
      print(j)
      delta = np.zeros((S.shape[0]))
      count = 0
      for y in range(S.shape[0]):
          k1 = 1
          k2 = gp_kernel1(sita,S[y,1:4],A[:,1:4])
          k3 = gp_kernel1(sita,A[:,1:4],A[:,1:4])
          delta[count] = k1 - np.matmul(k2, np.matmul(np.linalg.inv(k3),k2.T))
          if delta[count]<0.5/(np.pi*np.e): # Entropy less than 0
              print(j,'---->',count)
          count = count + 1
      a =np.argmax(delta)
      uncertainty = np.append(uncertainty,np.max(delta))
      A = np.vstack((A, S[a,:]))
      S= np.delete(S,a,0)
  ```
  - Mutual information Maximization
  ```Jupyter Notebook
    # Data: sensor_position_candidate [2162 nodals, 4 dimensions], sensor_position_all [3726 nodals, 4 dimensions]
    # Package: import numpy

    # Function: Kernel definition
    def gp_kernel1(sita,x,y):
        if np.sum(x.shape)<4:
            x = x[None,:]
        if np.sum(y.shape)<4:
            y = y[None,:]
        gpcovariance = np.zeros((x.shape[0],y.shape[0]))
        ry = 0.4292
        rxy1 = np.sqrt(x[:,0]**2+x[:,2]**2)
        rxy2 = np.sqrt(y[:,0]**2+y[:,2]**2)
        angle1 = np.arctan2(x[:,2],x[:,0])
        angle2 = np.arctan2(y[:,2],y[:,0])
        for j in range(y.shape[0]):
            for i in range(x.shape[0]):
                dy_ = np.sqrt((x[i,1]-y[j,1])**2+(rxy1[i]-rxy2[j])**2)
                dy__ = 2*ry*np.arcsin((dy_/2)/ry)

                angle = np.abs(angle1[i] - angle2[j])
                if angle > np.pi:
                    angle = 2*np.pi-angle
                dxz__ = (rxy1[i]+rxy2[j])/2*angle

                index = np.sqrt((dy__**2+dxz__**2)) 
                gpcovariance[i,j] = np.exp(-index**2/sita**2)
         return(np.around(gpcovariance,decimals=10))

    # Function: Optimal sensor position selection
    sita = 0.0175
    for i in range(S.shape[0]):
        count = 0
        for j in range(V.shape[0]):
            if S[i,0] ==V[j,0]:
                count = count + 1
        if count ==0:
            V= np.vstack((V, S[i,:]))
    print(V.shape)

    A = np.zeros((1,4))
    A_ = np.copy(V)
    
    uncertainty = []
    k_base1 = gp_kernel1(sita,S[:,1:4],S[:,1:4])
    for i in range(30):
        print(i)
        k_base2 = gp_kernel1(sita,A_[:,1:4],A_[:,1:4])
        k1 = 1
        k1_ = 1
        delta = np.zeros(S.shape[0])
        if i==0:
            k_base3=1
            for y in range(S.shape[0]):
                k2 = np.delete(np.copy(k_base1[y,:]),y)
                k3 = 1
                for j in range(A_.shape[0]):
                    if S[y,0]==A_[j,0]:
                        k2_ = np.delete(np.copy(k_base2[j,:]),j)
                        k3_ = np.delete(np.delete(np.copy(k_base2),j,axis=0),j,axis=1)
                        break
                d1 = (k1-np.dot(k2,k2.T)/(S.shape[0]-1))
                d2 = (k1_-np.dot(k2_,np.dot(np.linalg.inv(k3_),k2_.T)))
    #             if d1<0 or d2<0: # Entropy less than 0
    #                 print(j,'---->',count)
                if np.linalg.det(k3_) < 0:
                    print("not pd")
                delta[y] = d1/d2

        else:
            k_base3 = gp_kernel1(sita,A[:,1:4],A[:,1:4])
            for y in range(S.shape[0]):
                k2 = gp_kernel1(sita,S[y,1:4],A[:,1:4])
                k3 = np.copy(k_base3)
                for j in range(A_.shape[0]): 
                    if S[y,0] == A_[j,0]:
                        k2_ = np.delete(np.copy(k_base2[j,:]),j)
                        k3_ = np.delete(np.delete(np.copy(k_base2),j,axis=0),j,axis=1)
                        break
                d1 = (k1-np.dot(k2,np.dot(np.linalg.inv(k3),k2.T)))
                d2 = (k1_-np.dot(k2_,np.dot(np.linalg.inv(k3_),k2_.T)))
                delta[y] = d1/d2 
    #             if d1<0 or d2<0: # Entropy less than 0
    #                 print(j,'---->',count)
                if np.linalg.det(k3_) < 0 or np.linalg.det(k3)<0:
                    print("not pd k3_")

        a = np.argmax(delta)
        uncertainty = np.append(uncertainty,np.max(delta))
        if i == 0:
            A = S[a,:][None,:]
        else:
            A = np.vstack((A, S[a,:]))
        for j in range(A_.shape[0]):
            if A[A.shape[0]-1,0]==A_[j,0]:
                A_ = np.delete(A_,j,0)
                break
        S = np.delete(S, a, 0)
    np.savetxt("../02_Data_Processing/02_Mutual_Information/01_MI_Positions.txt",A,delimiter=' ')
    ```
With no hesitation to contact **huanbo.sunrwth@gmail.com** for unclear explaination. 
