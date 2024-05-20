import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import softmax
from scipy.spatial.distance import euclidean

# MS: TODO comment out for now
#import biomart


import umap
import pickle
import scipy.spatial as sp
import seaborn as sns
import itertools

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris

from scipy.spatial.distance import pdist,squareform
from scipy.cluster import hierarchy

from numpy.linalg import eig

import glob


class Visualizations(object):
    def plot_pca_trajs(ax,patterns,resolvers,highlight_idx=[]):
        pca = PCA(n_components=2)
        pca.fit(patterns)
        pca_data = pca.transform(patterns)
        ax.scatter(pca_data[:,0],pca_data[:,1],c='gray',s=20)
        for i,resolver in enumerate(resolvers):
            pca_traj = pca.transform(resolver.resolved_array)
            if i in highlight_idx:
                ax.plot(pca_traj[:,0],pca_traj[:,1],lw=3,c='r')
            else:
                ax.plot(pca_traj[:,0],pca_traj[:,1],lw=1,c='k')
        ax.grid(False)
        ax.set_facecolor('white')
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_yticks([])
        ax.set_xticks([])
        
        
    def label_heatmap_annealing(ax,arr):
        N,K = arr.shape
        ax.set_yticks(range(K))
        ax.set_xticks([0.01*N,N*.3,N*.6,N*.9])
        ax.set_xticklabels([r'high $\beta$',r'low $\beta$',r'med. $\beta$',r'high $\beta$'],rotation=90)
        ax.set_yticks(np.arange(-.5,K+.5),minor=True)
        ax.set_yticklabels([(i+1) for i in range(K)])
        
        
    def plot_enhancer_probs(ax,resolver,aspect=.5,label_annealing=True,cbar=False,vmax=0):
        enhancer_probs = np.array([(np.exp(beta*np.matmul(resolver.patterns,traj))/np.exp(beta*np.matmul(resolver.patterns,traj)).sum()) for traj,beta in zip(resolver.resolved_array,resolver.beta_vals)])

        cax=ax.imshow(enhancer_probs.T,aspect=aspect*enhancer_probs.shape[0]/enhancer_probs.shape[1],vmin=0,vmax=1,cmap="Greens")
        
        if vmax==0:
            vmax=enhancer_probs.max()
        
        if label_annealing:
            Visualizations.label_heatmap_annealing(ax,enhancer_probs)
        else:
            ax.set_yticks(np.arange(-.5,resolver.patterns.shape[0]+.5),minor=True)
            ticks=np.arange(0,resolver.resolved_array.shape[0]-1,100)
            ax.set_xticks(ticks)
            ax.set_xticklabels([int(np.round(x)) for x in resolver.t[ticks]])
        ax.grid(visible=True, which='major', color='gray',lw=0)
        ax.grid(visible=True, which='minor', color='gray',lw=1)
        ax.set_ylabel('enhancer')
        if cbar:
            plt.colorbar(ax=ax,mappable=cax)
        
    def plot_tf_levels(ax,resolver,aspect=.5,label_annealing=True,cbar=False,vmax=0):
        if vmax==0:
            vmax=resolver.patterns.max()
        cax=ax.imshow(resolver.resolved_array.T,aspect=aspect*resolver.resolved_array.shape[0]/resolver.resolved_array.shape[1],
                  vmin=0,vmax=vmax,cmap="Greens")
        
        if label_annealing:
            Visualizations.label_heatmap_annealing(ax,resolver.resolved_array)
        else:
            ax.set_yticks(np.arange(-.5,resolver.patterns.shape[1]+.5),minor=True)
            ticks=np.arange(0,resolver.resolved_array.shape[0]-1,100)
            ax.set_xticks(ticks)
            ax.set_xticklabels([int(np.round(x)) for x in resolver.t[ticks]])
        ax.grid(visible=True, which='major', color='gray',lw=0)
        ax.grid(visible=True, which='minor', color='gray',lw=1)
        ax.set_ylabel('TF')
        if cbar:
            plt.colorbar(ax=ax,mappable=cax)
            
    def umap_plot_diff_trajs(ax,resolvers):
        umap_patterns=[]
        for resolver in resolvers:
            umap_patterns.append(resolver.resolved_array[5::5])

        ln=len(umap_patterns[-1])
        umap_patterns = np.concatenate(umap_patterns)

        ft = umap.UMAP(n_neighbors=15).fit_transform(umap_patterns)

        colors = ['red','green','purple','blue','gold','gray','orange','#1f77b4', '#8c564b','#e377c2','#17becf']

        used_names = np.array([])

        for idx,resolver in zip(np.arange(0,ft.shape[0],ln),resolvers):
            name = resolver.tag
            if not name in used_names:
                ax.text(ft[idx+ln-1,0]-3,ft[idx+ln-1,1]+.5,name,fontsize=20)
                used_names = np.append(used_names,name)
            id_=np.argwhere(used_names==name)[0,0]
            ax.scatter(ft[idx:idx+ln,0],ft[idx:idx+ln,1],c=colors[id_])
            ax.plot(ft[idx:idx+ln,0],ft[idx:idx+ln,1],c=colors[id_],lw=.1)

        ax.set_xlabel("UMAP 1")    
        ax.set_ylabel("UMAP 2")    
        ax.grid(False)
        ax.set_facecolor('white')
        
    def plot_patterns(ax,patterns,aspect=.5):
        K,N=patterns.shape
        ax.imshow(patterns.max()-patterns.T,aspect=aspect*patterns.shape[0]/patterns.shape[1],
                  cmap='gray',vmin=0,vmax=patterns.max())
        ax.set_yticks(range(N))
        ax.set_xticks(range(K))
        ax.set_yticks(np.arange(-.5,N+.5),minor=True)
        ax.set_xticks(np.arange(-.5,K+.5),minor=True)
        ax.grid(visible=True, which='major', color='gray',lw=0)
        ax.grid(visible=True, which='minor', color='gray',lw=1)
        ax.set_xticklabels(['EN%d'%(i+1) for i in range(K)],rotation=90)
        ax.set_yticklabels(['TF%d'%(i+1) for i in range(N)],rotation=0)

    def plot_heatmap_dendrogram(enhancer_net,ax_heatmap,ax_dendrogram):
        def est_dendrogram(ax,enhancer_net):
            ydist = pdist(enhancer_net.patterns,metric='cosine')
            dist_mat = pd.DataFrame(squareform(ydist), index=enhancer_net.names, columns= enhancer_net.names)

            Z = hierarchy.linkage(100*ydist, 'average')
            dn = hierarchy.dendrogram(Z,labels=enhancer_net.names,
                                      orientation='left',leaf_rotation=0,
                                      color_threshold=0,above_threshold_color='k',
                                      leaf_font_size=0,ax=ax,show_leaf_counts=True)
            ax.set_yticks([])
            ax.set_xticks([])
            hierarchy.set_link_color_palette(None) 

            ax.grid(False)
            ax.set_facecolor('white')
            return dist_mat,dn['ivl']

        dist_mat,order = est_dendrogram(ax_dendrogram,enhancer_net)

        sns.set(font_scale=1.4)
        cg=sns.heatmap(dist_mat.loc[order[::-1],order[::-1]],annot_kws={"size": 30},cmap='crest',
                      cbar_kws=dict(orientation='vertical',location='right'),ax=ax_heatmap)
        ax_heatmap.set_xlabel("")
        ax_heatmap.set_ylabel("")
        ax_heatmap.collections[0].colorbar.set_label(r'$1-\cos{\theta}$')

          

class Resolver(object):
    def __init__(self,ximat,w=None,qmat=None):
        self.patterns = ximat.copy()
        if type(qmat)==type(None):
            qmat = self.patterns
        if type(w)==type(None):
            w=np.zeros(self.patterns.shape[0])
        self.qmat = qmat
        self.w = w
    def euler_step(self,x,beta,x_prod,sigma=0,dt=.1):
        det_dx = x_prod + np.matmul(self.qmat.T,softmax(self.w + beta*np.matmul(self.patterns,x))) - x
        if sigma>0:
            noise_dx = sigma*np.random.normal(size=x.shape[0])
        else:
            noise_dx = 0
            
        return dt*(det_dx)+np.sqrt(dt)*noise_dx
    
    def resolve(self,x0,tmax,beta_func,sigma=0,x_prod_func = None,dt=.1):
        """
        resolve initializes the values of resolved_array,beta_vals,and t after resolution by iteration
        """
        self.t = np.arange(0,tmax,dt)
        self.beta_vals = np.array([beta_func(t) for t in self.t])
        self.resolved_array = np.zeros([self.t.shape[0],len(x0)])
        
        if type(x_prod_func)==type(None):
            null_prod = np.zeros(x0.shape[0])
            x_prod_func = lambda t: null_prod
        x = x0.copy()
        for i,t in enumerate(self.t):
            self.resolved_array[i,:]=x.copy()
            x += self.euler_step(x,beta_func(t),x_prod_func(t),sigma,dt)
        return x,t,self.beta_vals,self.resolved_array
    
    def tag(self,tag):
        self.tag = tag
        
    def jacobian(self,beta,x):
        def est_jac_idx(j,k):
            activation_vec = np.exp(self.w + beta*np.dot(self.patterns,x))
            Z = activation_vec.sum()
            jac_A = beta * ((activation_vec*self.qmat[:,j]*self.patterns[:,k]).sum()/Z - (activation_vec*self.qmat[:,j]).sum()*(activation_vec*self.patterns[:,k]).sum() / (np.power(Z,2)))
            if not j==k:
                return jac_A
            else:
                return jac_A - 1
        jac = np.zeros((self.patterns.shape[1],self.patterns.shape[1]))
        for j in range(jac.shape[0]):
            for k in range(jac.shape[1]):
                jac[j,k] = est_jac_idx(j,k)
        return jac
    

class TFProcess(object):
    def initialize_patterns(df):
        patterns = (df.copy().values)
        norm_ = np.linalg.norm(patterns,axis=1)
        patterns = np.array([patterns[i]/norm_[i] for i in range(patterns.shape[0])])

        return patterns,list(df.index),list(df.columns)
    
    def __init__(self,df,w):
        if type(w)==type(None):
            w=np.zeros(df.shape[1])
        self.w = w
        self.patterns,self.names,self.TFs = TFProcess.initialize_patterns(df)
        
    def most_similar_x(self,res):
        return self.names[np.argmin([euclidean(pattern,res) for pattern in self.patterns])]
    
    def avg_state(self):
        return self.patterns.mean(axis=0)

class TFProcessStaticBeta(TFProcess):
    def __init__(self,df,w=None,beta=120):
        self.beta = beta
        TFProcess.__init__(self,df,w)
        
   
    def generate_noisy_initial_condition_sim(self,name,noise_mag = 0.2,duration=30):
        resolver = Resolver(self.patterns,self.w)
        idx = np.argwhere([x==name for x in self.names])[0,0]
        x0=self.patterns[idx]*np.random.normal(1,noise_mag,self.patterns.shape[1])
        x,t,_,resolved_array = resolver.resolve(x0,duration,lambda t: self.beta)
        return self.most_similar_x(resolved_array[-1]),resolver
        
    def reprogram(self,init_cell,factors,strengths,pulse_width=10):
        resolver = Resolver(self.patterns,self.w)
        
        activation_vec_orig = np.zeros(len(self.TFs))
        activation_vec_mod = activation_vec_orig.copy()
        activation_vec_mod[np.array([(TF in factors) for TF in self.TFs])] = strengths
        
        x0 = self.patterns[np.argwhere(np.array(self.names)==init_cell)[0,0]]
        def x_prod_func(t):
            if t<pulse_width:
                return activation_vec_orig
            elif t<2*pulse_width:
                return activation_vec_mod
            else:
                return activation_vec_orig
        resolver.resolve(x0,3*pulse_width,lambda t: self.beta,x_prod_func=x_prod_func)
        resolver.tag(self.most_similar_x(resolver.resolved_array[-1]))
        return resolver
    
    def process_reprogramming_input(self,reprog_paths,verbose=True):
        resolvers = []
        for i in range(reprog_paths.shape[0]):
            resolver = self.reprogram(reprog_paths.iloc[i].loc["source cell"],reprog_paths.iloc[i]["factors"].split(","),[float(x) for x in reprog_paths.iloc[i].loc["weights"].split(",")])
            resolvers.append(resolver)
            if verbose:
                print("PREDICTION: ",reprog_paths.iloc[i].loc["target cell"])
                print("RESULT:", result)
        return resolvers
    
    def generate_noised_input_trajectories(self,assert_same=True):
        resolvers = []
        for name in self.names:
            target,resolver = self.generate_noisy_initial_condition_sim(name,noise_mag=0.2)
            if assert_same:
                assert (name==target)
            resolvers.append(resolver)
        return resolvers
    
class TFProcessAnnealing(TFProcess):
    def __init__(self,df,w=None,sigma=0,beta_max=5,frac_init=0.1):
        self.frac_init = frac_init
        self.beta_max = beta_max
        self.sigma=sigma
        TFProcess.__init__(self,df,w)
        
    def annealing(self,x0,tmax=100):
        resolver = Resolver(self.patterns,self.w)
        
        tinit=tmax*self.frac_init
        tgap = tmax-tinit
        betafunc = lambda t: self.beta_max if t < tinit else .01 + self.beta_max*(t-tinit)/tgap
        
        resolver.resolve(x0,tmax,betafunc,sigma=self.sigma)
        resolver.tag(self.most_similar_x(resolver.resolved_array[-1]))
        return resolver
    
    def adj_w_balanced_diff(self,max_iter=1000):
        x0=self.avg_state()
        w=pd.Series(np.zeros(self.patterns.shape[0]),index=self.names)
        hist_=[]
        for iter_ in range(max_iter):
            self.w=w.values
            resolver = self.annealing(x0)
            j = self.most_similar_x(resolver.resolved_array[-1])
            w[j]-=.5*(1-iter_/max_iter)
            w=w-w.mean()
        self.w=w

    def produce_balanced_differentiation_trajectories(self,traj_output_number=20,max_iter_=1000,verbose=True):
        self.adj_w_balanced_diff()
        hist=np.array([])
        resolvers=[]
        for iter_ in range(max_iter_):
            resolver=self.annealing(self.avg_state())
            hist = np.append(hist,resolver.tag)
            if (hist==resolver.tag).sum()<traj_output_number+1:
                resolvers.append(resolver)
            
        return resolvers


class Genome(object):
    """
    The Genome object is an object that contains an explicit representaiton of all enhancers and associations in a genome
    each enhancer binds one of N tfs, indexed 1,...,N
    """
    def __init__(self):
        self.xi_mat = None
        self.q_mat_T = None
        self.enhancer_ids = []
    def add_enhancer(self,tf_binding_dict,q_association_dict,normalize=False,eid=-1):
        """
        TF binding dict = format {TF : XI_TF}
        Q binding dict = format {TF : Q_TF}
        """
        xi_vec = np.zeros(self.N)
        q_vec = np.zeros(self.N)
        for tf in tf_binding_dict.keys():
            xi_vec[tf] = tf_binding_dict[tf]
        for tf in q_association_dict.keys(): 
            q_vec[tf]  = q_association_dict[tf]
            
        if normalize:
            xi_vec = xi_vec /np.linalg.norm(xi_vec)
        
        if type(self.xi_mat)==type(None):
            self.xi_mat = xi_vec.reshape(1,self.N)
        else:
            self.xi_mat = np.append(self.xi_mat,xi_vec.reshape(1,self.N),axis=0)
        if type(self.q_mat_T)==type(None):
            self.q_mat_T = q_vec.reshape(self.N,1)
        else:
            self.q_mat_T = np.append(self.q_mat_T,q_vec.reshape(self.N,1),axis=1)
            
        self.enhancer_ids.append(eid)
            
    def run_x0(self,x0,tmax=10,beta=50,w=None):
        resolver = Resolver(ximat=self.xi_mat,qmat=self.q_mat_T.T)
        resolver.resolve(x0,tmax,lambda t: beta)
        
        return resolver
        
        
class GenomeFromTFDicts(Genome):        
    def __init__(self,tf_dicts,q_dicts):
        # N - number of TFs
        self.N = np.max([np.max(x) for x in q_dicts.values()]) + 1
        Genome.__init__(self)
        
        for tf_dict,q_dict in zip(tf_dicts,q_dicts):
            self.add_enhancer(tf_dict,q_dict)
            
class GenomeFromEnAssociations(Genome):
    def __init__(self,enhancer_binding_profiles,enhancer_locations):
        self.N = np.max([np.max(x) for x in enhancer_locations.values()]) + 1
        Genome.__init__(self)
        
        for enhancer in enhancer_binding_profiles.keys():
            for location in enhancer_locations[enhancer]:
                q_dict = {location : 1}
                tf_dict = dict([(tf,1) for tf in enhancer_binding_profiles[enhancer]])
                self.add_enhancer(tf_dict,q_dict)