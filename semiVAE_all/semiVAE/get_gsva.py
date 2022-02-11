#from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
import rpy2.robjects as ro
import pandas as pd
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

def get_gsva(sample_id, file_exp, file_gene_sig):
    r = robjects.r
    r['source']('semiVAE/gsva.r')
    gsva_fun = robjects.globalenv['gsva_fun']
    gsva_result_r = gsva_fun(sample_id, file_exp, file_gene_sig)
    with localconverter(ro.default_converter + pandas2ri.converter):
        gsva_result = ro.conversion.rpy2py(gsva_result_r)
    
    gsva_result = pd.DataFrame(gsva_result,
                               columns=pd.read_csv(file_exp,index_col=0).index,
                               index=pd.read_csv(file_gene_sig).columns
                              ).transpose()
    return gsva_result

