from linearmodels import IV2SLS, IVLIML, IVGMM, IVGMMCUE
import pandas as pd
import numpy as np


class PanelLaggedDep(IVGMM):
    '''
    Estimates values of rho and beta in the following Arrelano Bond model on a panel data set:
    
    y(i,t) = rho*y(i,t-1) + beta*x(i,t) + u(i) + e(i,t)
    
    Where y is the observed endogenous variable (scalar), x is an observed vector of exogenous variables, 
    u is an unobserved individual effect, and e(i,t) is independent across i and t.
    
    Fixed effects OLS will usually result in biased estimates of rho because of the correlation between 
    y(i,t) and e(i,t) stemming from u(i). The Arrelano-Bond procedure first differences the observables 
    and then uses lags of y as instruments.
    
    The `endog` and `exog` vars are assumed to have a multiindex structure with the first level identifying 
    cross-section and the second identifying time. The time level must be sorted in ascending order. The
    panel data can have missing values and be unbalanced (i.e., different sample size for each group).
    
    `lags` is the number of lags of the `endog` series. Untested for anything other than lags==1
    '''
    def __init__(self, endog, exogs, lags=1, iv_max_lags=1000, systemGMM = False, add_intercept = False):
        min_t = min(endog.index.get_level_values(1)) #starting period
        T = max(endog.index.get_level_values(1)) + 1 - min_t #total number of periods
        
        #Names for the original and differenced endog and exog vars
        ename = endog.name
        Dename = 'D' + ename
        xnames = exogs.columns.tolist()
        Dxnames = ['D'+x for x in xnames]

        #We'll store all of the data in a dataframe
        self.data = pd.DataFrame()
        self.data[ename] = endog
        self.data[Dename] = endog.groupby(level=0).diff()

        #Generate and store the lags of the differenced endog variable
        Lenames = []
        LDenames = []
        for k in range(1,lags+1):
            col = 'L%s%s'%(k,ename)
            colD = 'L%s%s'%(k,Dename)
            Lenames.append(col)
            LDenames.append(colD)
            self.data[col] = self.data[ename].shift(k)
            self.data[colD] = self.data[Dename].shift(k)
        
        #Store the original and the diffs of the exog variables
        for x in xnames:
            self.data[x] = exogs[x]
            self.data['D'+x] = exogs[x].groupby(level=0).diff()

        #Set up the instruments -- lags of the endog levels for different time periods
        instrnames = []
        gb = endog.groupby(level=0)
        for k in range(lags+1, min(T, lags+1+iv_max_lags)): #TODO: Check this -- works for lags == 1, not sure if for anything else
            shifted = gb.shift(k)
            for t in range(k, T):
                col = 'ILVL_t%iL%i'%(t+min_t,k)
                instrnames.append(col)
                instrument = pd.Series(0, index=endog.index)
                data_pos = endog.index.get_level_values(1) == t+min_t
                instrument.loc[data_pos] = shifted[data_pos]
                self.data[col] = instrument

        self.data[[ename] + instrnames].to_csv('instruments.csv')

        if systemGMM:
            #With the systems GMM estimator we have additional instruments of lagged differences
            instrnamessys = []
            for t in range(lags, T): #TODO: Check this -- works for lags == 1, not sure if for anything else
                col = 'IDIFF_t%iL'%t
                instrnamessys.append(col)
                self.data[col] = self.data[LDenames[0]].groupby(level=0).shift(lags)
                self.data.loc[endog.index.get_level_values(1) != t+min_t, col] = 0

            #Then to estimate the system GMM we stack differenced and undifferenced data and their corresponding instruments
            cols1 = [Dename]+LDenames+Dxnames+instrnames #variables used for the differences part of the regression
            cols2 = [ename]+Lenames+xnames+instrnamessys #variable used for the levels part of the regression
            cols2R = [Dename]+LDenames+Dxnames+instrnamessys #used to rename the variables that we want to overlap in the system reg.

            #The differenced data set
            data1 = self.data[cols1].copy()
            for c in instrnamessys:
                data1[c] = 0 #zero out the instruments that apply to undifferenced data
            
            #The undifferenced data set
            data2 = self.data[cols2].copy()
            data2.columns = cols2R
            for c in instrnames:
                data2[c] = 0 #zero out the instruments that apply to differenced data
            
            #Add an index level to each data series so we can join without creating duplicate index values
            a1 = data1.index.get_level_values(0)
            a2 = data1.index.get_level_values(1)
            n = len(a1)
            data1.index = pd.MultiIndex.from_arrays([[1]*n,a1,a2])
            data2.index = pd.MultiIndex.from_arrays([[2]*n,a1,a2])
            
            #Now append the series together
            self.data = data1.append(data2)
                
        # Add intercept
        if add_intercept:
            self.data['constant'] = 1
            Dxnames += ['constant']
            
        dropped = self.data.dropna()
        dropped['CLUSTER_VAR'] = dropped.index.get_level_values(0)
        IVGMM.__init__(self, dropped[Dename], dropped[Dxnames], dropped[LDenames], dropped[instrnames], weight_type='clustered', clusters = dropped['CLUSTER_VAR'])
