# ml_Adams_Le

#Necessary Import Downloads to run 


pip install torch torchtext scikit-learn pandas psutil transformers

torch and torchtext specifically need to be installed with certain versions to ensure no conflict between them. 

For us, it worked with torch 2.2.2+cpu and torchtext 0.17.2+cpu. 

For the model files, merely change the number for the intended dataset. 

One near the top, 
df = pd.read_csv('datasets/WELFake_sub_dataset_3.csv')

and one near the bottom,
df.to_csv("datasets/cnn2_dataset3.csv", index=False)
