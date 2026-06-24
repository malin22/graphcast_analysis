## Towards a Mechanistic Understanding of GraphCast Through Latent Analysis 

In this repo we provide the code for investigating the internal latent activations of GraphCast using principal component analysis (PCA) 
applied to the processor-layer embeddings across multi-year global forecasts. By projecting latent activations onto 
principal component directions and correlating the resulting spatial activation maps with ERA5 physical fields, we 
explore the extent to which physically interpretable atmospheric structure is linearly represented within GraphCast’s 
latent space. 

The core codebase is in /src with setting up the data and graphcast and running PCA.

In /sabines_*_experiments and /malins_experiments we currently have first drafts of scripts 
that are currently being worked on.

Results are in /plots and the data is held seperately on the Uni Osnabrück HPC the shared folder at /share/prj-4d/graphcast_shared/data. 



