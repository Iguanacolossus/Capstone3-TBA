import numpy as np
#rotatation around z axis
def rot_matrix_z(x,y,z,theta):
    '''performs a theta degree rotation around the x axis of any three demetional point in space'''
    '''inputs:
            x : <float, y: <float>,  z: <float>
            theta: degrees
       output:
           new x y z 
           '''
    theta_rad = theta/57.2958
    rot_matrix = np.array([[np.cos(theta),-np.sin(theta), 0],
                               [np.sin(theta),np.cos(theta), 0],
                              [0, 0, 1]])
    
    return np.array([x,y,z])@rot_matrix

# rotate the three x,y,z arrays and save them in new x,y,z arrays
def rotate_xyz(x_array,y_array,z_array,theta):
    x_rot = np.array([])
    y_rot = np.array([])
    z_rot = np.array([])
    for i in range(len(x_array)):
        new_point = rot_matrix_z(x=x_array[i],y=y_array[i],z=z_array[i],theta=theta)
        x_rot = np.append(x_rot,new_point[0])
        y_rot = np.append(y_rot,new_point[1])
        z_rot = np.append(z_rot,new_point[2])
    return x_rot,y_rot,z_rot

##Convert to cartesian points to scherical points seperatly
def to_rho(x,y,z):
    rho = np.sqrt(x**2 + y**2 + z**2)
    return rho

def to_theta(x,y):
    return np.arctan(y/x)

def to_phi(x,y,z):
    return np.arccos(z/to_rho(x,y,z))

## convert x,y,and z arrays to rho,theta, phi arrays
def cart_to_polar(x_array,y_array,z_array):
    '''input three array representing x y and z
        out put three arrays representing rho,theta, phi'''
    rho = np.array([])
    theta = np.array([])
    phi = np.array([])
    for i in range(len(x_array)):
        rho = np.append(rho,to_rho(x_array[i],y_array[i],z_array[i]))
        theta = np.append(theta, to_theta(x_array[i],y_array[i]))
        phi = np.append(phi,to_phi(x_array[i],y_array[i],z_array[i]))
    return rho,theta,phi

def df_to_rho(df):
    '''converts pandas df of x,y,z series to just rho series'''
    rho_df = np.empty((0,206))
    
    ### for each row
    for row in df.iterrows():
        
        ### create x y z from each row
        x_array= list(row[1][0][0])
        x_array = np.array(x_array)
        x_array = np.reshape(x_array,(1,-1))#[0]
        
        y_array = list(row[1][0][1])
        y_array = np.array(y_array)
        y_array = np.reshape(y_array,(1,-1))#[0]
        
        z_array = list(row[1][0][2])
        z_array = np.array(z_array)
        z_array = np.reshape(z_array,(1,-1))#[0]
        
        #make a new sample that is just rho
        rho,theta,phi = cart_to_polar(x_array,y_array,z_array)
        rho = np.reshape(rho,(1,-1))
       #creat new df convertes to just the rho element of polar coordinanace
        
        rho_df = np.append(rho_df, rho,axis=0)

    return rho_df

def df_to_rho_phi(df):
    '''Converts the who pandas dataframe of lists if lists if cartesian coordiance 
    to numpy aray of the rho and phi components of the origonal pd dataframe
        input: pandas df of nested lists
        output: numpy array with lists as channels (3rd dimention)'''
    rho_df = np.empty((0,206))
    phi_df = np.empty((0,206))
    
    ### for each row
    for row in df.iterrows():
        
        ### create x y z from each row
        x_array= list(row[1][0][0])
        x_array = np.array(x_array)
        x_array = np.reshape(x_array,(1,-1))#[0]
        
        y_array = list(row[1][0][1])
        y_array = np.array(y_array)
        y_array = np.reshape(y_array,(1,-1))#[0]
        
        z_array = list(row[1][0][2])
        z_array = np.array(z_array)
        z_array = np.reshape(z_array,(1,-1))#[0]
        
        #convert sample to shoereical coordinance
        rho,theta,phi = cart_to_polar(x_array,y_array,z_array)
        rho = np.reshape(rho,(1,-1))
        phi = np.reshape(phi,(1,-1))

       #creates a rho df and phi df
        rho_df = np.append(rho_df, rho,axis=0)
        phi_df = np.append(phi_df,phi,axis=0)
        
            # stacks the rho and phi dataframs creating channels
    return np.dstack((rho_df,phi_df))

def intra_cluster_dist(X,labels,metric,i):
    '''Calculate the mean intra-cluster distance for sample i.
     Returns
    -------
    a : float
        Mean intra-cluster distance for sample i
    '''
    indices = np.where(labels == labels[i])[0]
    
    if len(indices) == 0:
        return 0.
    a = np.mean([metric(X[i], X[j]) for j in indices if not i == j])
    return a

def nearest_cluster_distance(X,labels,metric,i):
    '''Calculate the mean nearest-cluster distance for sample i.'''
    '''    Returns
    -------
    b : float
        Mean nearest-cluster distance for sample i'''
    label = labels[i]
    b = np.min(
            [np.mean(
                [metric(X[i], X[j]) for j in np.where(labels == cur_label)[0]]
            ) for cur_label in set(labels) if not cur_label == label])
    return b

def sil_samples_2(X,labels,metric):
    '''Compute the Silhouette Coefficient for each sample.'''
    n = labels.shape[0]
    A = np.array([intra_cluster_dist(X, labels, metric, i)
                  for i in range(n)])
    B = np.array([nearest_cluster_distance(X, labels, metric, i)
                  for i in range(n)])
    sil_samples = (B - A) / np.maximum(A, B)
    # nan values are for clusters of size 1, and should be 0
    return np.nan_to_num(sil_samples)

from itertools import compress
def make_sil_plot(x,y,metric,fitted_model):
    '''makes sillhouette plots
    INPUT: x = featur matrix
           y = cluter predictions
           metric = distance metric
           fitted_model = fitted cluterin gmodel
           
    OUTPUT: no return just makes plot
    '''
    cluster_labels = np.unique(y)
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = sil_samples_2(x,y,metric)
    y_ax_lower, y_ax_upper = 0,0
    yticks = []
    for i ,c in enumerate(cluster_labels):
        c_silhouette_vals = list(compress(silhouette_vals,y==c))

        c_silhouette_vals = sorted(c_silhouette_vals)
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i)/n_clusters)
        plt.barh(range(y_ax_lower,y_ax_upper),
                c_silhouette_vals,
                height=1.0,
                edgecolor='none',
                color=color)
        yticks.append((y_ax_lower + y_ax_upper)/2.)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg,
               color='red',
               linestyle='--')
    plt.yticks(yticks,cluster_labels +1)
    plt.ylabel('Cluster',size=14)
    plt.xlabel('Silhouette coefficient',size=14)
    #plt.savefig('../images/sil_plot_c3.png')  

def lr_plot(history_obj,model_name,file_name):
    ''' input : model , model_name <str> :file name for saving <str>'''
    history = history_obj.history
    x_arr = np.arange(len(history['loss']))+1
    fig = plt.figure(figsize=(12,4))
    ax= fig.add_subplot(1,2,1)
    ax.plot(x_arr,history['loss'],'-',label='train')
    ax.plot(x_arr,history['val_loss'],'-',label='validation')
    ax.legend()
    ax.set_xlabel('Epoch',size=15)
    ax.set_ylabel('Loss',size=15)

    ax = fig.add_subplot(1,2,2)
    ax.plot(x_arr,history['recall'],'-',label='train')
    ax.plot(x_arr,history['val_recall'],'-',label='val')
    ax.legend()
    ax.set_xlabel('Epochs',size=15)
    ax.set_ylabel('Recall',size=15)
    fig.suptitle(model_name,fontsize='16')
    #plt.savefig(f'../images/{file_name}')


def stack_channels(im1, im2):
    '''stacks to images on top of each other'''
    return np.dstack((im1,im2))

def ts_df_to_spectrogram(ts_df, channels=2):
    '''input is a 3 demetional dataframe of time series data (N,206,2)
    outout is a 4d array of spectrograms (N,11,11,2)'''
    spec_df = np.empty((11,11,channels))
    list_ = []
    for sample in range(ts_df.shape[0]):
        freqs, times, Sx1 = signal.spectrogram(X_train_rhophi[sample][:,0], nperseg=20,noverlap=2) 
        freqs, times, Sx2 = signal.spectrogram(X_train_rhophi[sample][:,1], nperseg=20,noverlap=2)
        im = stack_channels(Sx1,Sx2)
        
        list_.append(im)
        spec_df = np.stack(list_, axis=0)
    return spec_df