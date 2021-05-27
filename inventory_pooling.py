import numpy as np
def pooling_benefit(demands_mu, demands_sig, correlations=None, service_level=0.95):
    from scipy.stats import norm
    n=len(demands_mu); z=norm.ppf(service_level)
    ss_separate=sum(z*s for s in demands_sig)
    if correlations is None:
        pooled_sig=np.sqrt(sum(s**2 for s in demands_sig))
    else:
        cov=np.outer(demands_sig,demands_sig)*np.array(correlations)
        pooled_sig=np.sqrt(np.sum(cov))
    ss_pooled=z*pooled_sig
    reduction=1-ss_pooled/ss_separate if ss_separate>0 else 0
    return {"ss_separate":round(ss_separate,0),"ss_pooled":round(ss_pooled,0),
            "reduction_pct":round(reduction*100,1),"sqrt_law_approx":round((1-1/np.sqrt(n))*100,1)}
if __name__=="__main__":
    mu=[100,120,80,90,110]; sig=[25,30,20,22,28]
    print("Independent:",pooling_benefit(mu,sig))
    corr=np.eye(5); corr+=0.3*(1-np.eye(5))
    print("Correlated (0.3):",pooling_benefit(mu,sig,corr.tolist()))
