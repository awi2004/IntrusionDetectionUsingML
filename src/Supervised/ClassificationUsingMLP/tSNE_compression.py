from sklearn.manifold import TSNE
import numpy as np
import argparse




if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Xsrc", help="src address of X test")
    parser.add_argument("--Xname", help="file name")
    args = parser.parse_args()
    X = np.loadtxt(args.Xsrc)
    print("data loaded")
    tsne = TSNE(n_components=2,perplexity=100, random_state=0,learning_rate=100,n_iter=1500)
    X_2d = tsne.fit_transform(X)
    np.save(args.Xname,X_2d)



