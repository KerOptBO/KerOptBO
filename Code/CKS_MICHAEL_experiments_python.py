Experiment(description='CKS experiment',
           data_dir='NIPS_2024/synthetic/MICHAEL/CKS/michael.mat',
           results_dir='NIPS_2024/synthetic/MICHAEL/CKS/',
           max_depth=10,                
           k=1,                         
           n_rand=5,                    
           iters=100,                   
           base_kernels='SE,Per,Lin,RQ,MAT',
           verbose=False,
           make_predictions=False,
           skip_complete=True)
