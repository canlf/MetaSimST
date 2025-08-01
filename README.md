# MetaSimST
We introduce MetaSimST, a unified framework that integrates meta-learning with self-supervised contrastive learning. Centered on a SimSiam-based architecture, MetaSimST employs both continuous and random masking strategies to extract robust low-dimensional representations of gene expression without relying on negative samples. Additionally, we develop an enhanced gene expression modeling module that adaptively incorporates spatial coordinates, histological image features, and transcriptomic information to facilitate more effective multimodal representation learning.


<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/1dc44482-eb4b-4945-ab83-85cf579b39aa" />

# Requirements
python == 3.9  
torch == 1.13.0  
scanpy == 1.9.2  
anndata == 0.8.0  
numpy == 1.22.3

# Dataset
(1) Human DLPFCs within the spatialLIBD at http://spatial.libd.org/spatialLIBD.  
(2) Mouse embryo data at https://db.cngb.org/stomics/mosta.  
(3) Human breast cancer at https://www.10xgenomics.com/datasets.  
(4) Mouse olfactory bulb datasets at https://db.cngb.org/stomics/data/

# Run
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

os.environ["R_HOME"] = r"D:\R\R-4.4.1"

dataset = '151673'

n_clusters = 7

adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)  
adata.var_names_make_unique()  
adata = adata_preprocess(adata)  
adata.obsm['feat'] = pca(adata, n_comps=50)  
adata = construct_interaction(adata)  


from MetaSimST import my_model  
model = my_model.SpatialTranscriptomicsModel(input_dim=adata.obsm['feat'].shape[1]).to(device)  
model.load_state_dict(torch.load('model_state_dict1.pth')) 


from graph_func import preprocess_adj  
adj = preprocess_adj(adata.obsm['adj'])  
adj = torch.FloatTensor(adj).to(device)  
x_input = torch.tensor(adata.obsm['feat']).to(device)  
embedding = model.encoder(x_input, adj)  
adata.obsm['emb'] = embedding.detach().cpu().numpy() 


clustering(adata, n_clusters, key='emb', method=mulrst, refinement=True)   
df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')  
df_meta_layer = df_meta['layer_guess']  
adata.obs['ground_truth'] = df_meta_layer.values  
adata = adata[~pd.isnull(adata.obs['ground_truth'])]  
ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth']) 


sc.pl.spatial(  
    adata,  
    img_key="hires",    
    color=["ground_truth", "domain"],    
    title=["Ground truth", "ARI=%.4f"%ARI],  
    show=True,  
)  

# Code
We propose MetaSimST, a unified framework that combines meta-learning with self-supervised contrastive learning.
The code will be made public after the thesis is accepted
