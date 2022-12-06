#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[7]:


def write_sweep_ids():
    sweep_path = "/u/dbeaglehole/tabular-benchmark/launch_config/sweeps/benchmark_sweeps.csv"
    
    sweep_ids=[]
    with open(sweep_path,"r") as f:
        for line in f.readlines()[1:]:
            sweep_id = line.split(",")[0]
            sweep_ids.append(sweep_id)
    
    new_path = "/u/dbeaglehole/tabular-benchmark/sweep_ids.txt"
    with open(new_path,"w") as f:
        # datasets=("OnlineNewsPopularity" "parkinsons" "superconduct" "energy")
        f.write("(")
        for sweep_id in sweep_ids:
            f.write('"' + sweep_id + '"')
            f.write(" ")
        f.write(")")


# In[8]:


write_sweep_ids()


# In[ ]:




