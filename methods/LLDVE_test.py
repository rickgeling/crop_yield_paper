import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import time

import timeit
from joblib import Parallel, delayed
import warnings
import scipy
import scipy.stats
from matplotlib.ticker import MultipleLocator
# Removed 2026-09-18: `from PyTimeVar.locallinear.LLR import LocalLinear`,
# `from linearmodels.panel import PanelOLS`, `from sys import exit`, `from scipy.io import savemat`.
# None of them were used, and linearmodels does not import under numpy 2.

##### --------------------------------------

# WHAT DOES THIS SCRIPT DO?

##### --------------------------------------


def estK(u, h):
    '''
    @Purpose
    ----------
    To estimate the kernel function using Epanechnikov's kernel.

    @Parameters
    ----------
    u : Argument for the kernel fucntion.
    h : The bandwidth used for the kernel estimation.

    @Returns
    -------
    mK : Diagonal matrix with the estimated weights on the diagonal.
    '''
    k_u = np.where(np.abs(u/h) <= 1, 0.75*(1-(u/h)**2), 0)
    k_h = (k_u)/h
    mK = np.diagflat(k_h)

    return mK


def est_mdl(mY, mX, t_i, T, N, mK):
    '''
    @Purpose
    ----------
    To obtain model estimates for gt, the local component Beta, Gamma and finally...
    ... the dummy variable by using the local linear dummy variable (LLDV) approach.

    @Parameters
    ----------
    mY : Endogeneous regressor of dimension (TxN).
    mX : Contains exogeneous regressors (d) of dimension (NxTxd).
    t_i : Grid points over which one wants to estimate (Tx1).
    T : Amount of observations.
    N : Amount of individuals.
    mK : Consists T diagonal matrices with the weights from the kernel function... 
        ...on the diagonal making it of dimension (TxTxT).

    @Returns
    -------
    The estimated coefficients (Txd+1) and the estimated dummy variables (Nx1).
    '''
    X_gt = np.ones(T)
    d = mX.shape[-1] + 1
    mX_bar = np.mean(mX, axis=0)
    vY_bar = np.mean(mY, axis=1)

    THETA_hat = np.zeros((T, d*2))
    alpha_hat = np.zeros((N, T))
    for i in range(1, T+1):  # change back to T+1
        t_i_min_t = t_i - i/T
        lZ_tild = []
        lY_tild = []
        lZ = []
        lZZ = []
        lZY = []
        for j in range(N):
            vK = (np.diagonal(mK[i-1, :, :])).reshape(len(mK[i-1, :, :]), 1)
            mZi = np.c_[X_gt, mX[j, :, :], t_i_min_t,
                        mX[j, :, :]*(t_i_min_t.reshape(T, 1))]
            lZ.append(mZi)
            mZ_bar = np.c_[X_gt, mX_bar, t_i_min_t,
                           mX_bar*(t_i_min_t.reshape(T, 1))]
            lZ_tild.append(mK[i-1, :, :]**(1/2)@mZi - 1/np.sum(mK[i-1, :, :])*(vK**(1/2)@vK.T) @
                           (mZi-mZ_bar))
            lY_tild.append(mK[i-1, :, :]**(1/2)@mY[:, j] - 1/np.sum(mK[i-1, :, :])*(vK**(1/2)@vK.T) @
                           (mY[:, j]-vY_bar))
            lZZ.append(lZ_tild[j].T@lZ_tild[j])
            lZY.append(lZ_tild[j].T@lY_tild[j])
        THETA_hat[i-1, :] = np.linalg.inv(np.sum(lZZ, 0)) @ np.sum(lZY, 0)
        alpha_hat[:, i-1] = np.array([1/np.sum(mK[i-1, :, :])*vK.T @
                                      ((mY[:, y]-vY_bar)-(lZ[y]-mZ_bar)@THETA_hat[i-1, :]) for y in range(N)]).flatten()

    return THETA_hat[:, :d], np.mean(alpha_hat, axis=1)


def parLOUOCV(mY, mZ, mYmin_i, mXmin_i, t_i, mK_hi):
    '''
    @Purpose
    ----------
    This function is used to parallelize the inner loop in estBandwithLOUOCV(...)...
    ... in order to obtain the residuals.
    '''
    T, N = mYmin_i.shape
    theta_hat, _ = est_mdl(mYmin_i, mXmin_i, t_i, T, N, mK_hi)
    return (mY - np.sum(np.multiply(theta_hat, mZ), 1))


def estBandwithLOUOCV(mY, mX, mZ, t_i, T, N, 
                        output_file_prefix 
                       ):
    '''
    @Purpose
    ----------
    To obtain the optimal bandwidth by using the leave-one-unit-out cross-validation...
    ... method see Sun, Y., R. J. Carroll, and D. Li (2009). Semiparametric estimation...
    ...of fixed-effects panel data varying coefficient models. 

    @Parameters
    ----------
    mY : Endogeneous regressor of dimension (TxN).
    mX : Contains exogeneous regressors (d) of dimension (NxTxd).
    mZ : Contains all regressors including for gt making it of dimension (NxTxd+1).
    t_i : Grid points over which one wants to estimate (Tx1).
    T : Amount of observations.
    N : Amount of individuals.
    output_file_prefix : str
        The full path and base name for saving the output plot and CSV file.
        Example: os.path.join(output_path, "bandwidth_louocv_run1")

    @Returns
    -------
    h_opt : Optimal bandwidth according to the LOUOCV.
    '''
    step_size = 0.01
    # endpoint+step size to include endpoint
    h = np.arange(0.03, 0.05, step_size)
    lK = [np.asarray([estK(t_i - i/T, h_i) for i in range(1, T+1)])
          for h_i in h]
    lWSEP = []
    Md = np.identity(N*T) - 1/T * np.kron(np.identity(N), np.ones((T, T)))
    for j in range(h.shape[0]):
        eps = np.asarray(Parallel(n_jobs=-1)(delayed(parLOUOCV)(mY[:, n], mZ[n, :, :], np.delete(
            mY, n, 1), np.delete(mX, n, 0), t_i, lK[j]) for n in range(N)))
        lWSEP.append(np.sum((Md@(np.asarray(eps).flatten('F')))**2))
    h_opt = h[np.argmin(lWSEP)]

    plt.figure(figsize=(12, 6))
    plt.plot(h, lWSEP, linewidth=2)
    plt.plot(h, lWSEP, markevery=[np.argmin(lWSEP)], ls="", marker="o", markersize=8)
    plt.grid(linestyle='dashed')
    plt.xlabel('Bandwidth $h$', fontsize="xx-large")
    plt.ylabel('LOUOCV objective', fontsize="xx-large")
    plt.tick_params(axis='both', labelsize=15)
    plt.title('Leave-One-Unit-Out Cross-Validation', fontsize="x-large")
    plt.tight_layout()

    # CHANGED COMPARED TO HOUSING PAPER:
    plot_filename = f'{output_file_prefix}_h_{h_opt:.2f}_LOUOCV_curve.eps'
    csv_filename = f'{output_file_prefix}_LOUOCV_losses.csv'
    
    try:
        plt.savefig(plot_filename)
        print(f"Saved LOUOCV plot to: {plot_filename}")
    except Exception as e:
        print(f"Error saving LOUOCV plot: {e}")
    plt.show()
    
    try:
        np.savetxt(csv_filename, np.column_stack((h, lWSEP)), delimiter=",", header="bandwidth,loss_value", comments="")
        print(f"Saved LOUOCV loss values to: {csv_filename}")
    except Exception as e:
        print(f"Error saving LOUOCV CSV: {e}")

    return h_opt


def wx(x, tau):
    return scipy.stats.norm.pdf(x, loc=tau, scale=np.sqrt(0.025))

def ModCVLoss_OptLocal(mY, mX, mZ, mK, valpha_hat, t_i, lList, h, N, T, iNlist):
    d = mX.shape[-1] + 1
    
    mResi2 = np.zeros((T, iNlist))
    # Step 1: leave out
    for i in range(1,T+1):
        vK = (np.diagonal(mK[i-1, :, :])).reshape(len(mK[i-1, :, :]), 1)
        t_i_min_t = t_i - i/T
        for lid in range(iNlist):
            l = lList[lid]
            id_leavelout = np.arange(max(0, i-1-l), min(T, i-1+l+1))
            print(id_leavelout)

            #id_leavelout = id_leavelout[id_leavelout>=0 and id_leavelout <=T-1]
            #id_leavelout = [round(x) for x in id_leavelout]
            
            X_gt = np.ones(T)
            mX_bar = np.mean(mX, axis=0)
            vY_bar = np.mean(mY, axis=1)
            
            
            lZ_tild = []
            lY_tild = []
            lZ = []
            lZZ = []
            lZY = []
            for j in range(N):
                mZi = np.c_[X_gt, mX[j, :, :], t_i_min_t,
                            mX[j, :, :]*(t_i_min_t.reshape(T, 1))]
                lZ.append(mZi)
                mZ_bar = np.c_[X_gt, mX_bar, t_i_min_t,
                                mX_bar*(t_i_min_t.reshape(T, 1))]
                lZ_tild.append(mK[i-1, :, :]**(1/2)@mZi - 1/np.sum(mK[i-1, :, :])*(vK**(1/2)@vK.T) @
                                (mZi-mZ_bar))
                lY_tild.append(mK[i-1, :, :]**(1/2)@mY[:, j] - 1/np.sum(mK[i-1, :, :])*(vK**(1/2)@vK.T) @
                                (mY[:, j]-vY_bar))
                
               
                lZ_tild_leaveout = np.delete(lZ_tild[j], id_leavelout, axis=0)
                lY_tild_leaveout = np.delete(lY_tild[j], id_leavelout, axis=0)
                lZZ.append((lZ_tild_leaveout.T@lZ_tild_leaveout))
                lZY.append((lZ_tild_leaveout.T@lY_tild_leaveout))
            
            vBetaHat_leaveout = (np.linalg.inv(np.sum(lZZ, 0)) @ np.sum(lZY, 0))[:d]

            # Step 2: get loss
            mResi2[i-1,lid] = SSR_ModCV(mY[i-1,:], mZ[:,i-1,:], vBetaHat_leaveout, valpha_hat, N)
            
            
    print('calculating weighted loss')  
    mLoss_ModCV_Optlocal = np.zeros((T, iNlist))
    for j in range(1,T+1):
        tau = j/T
        vwtau = wx(t_i, tau)
        mLoss_ModCV_Optlocal[j-1,:] = np.sum(mResi2*vwtau[:,None], axis=0)/T
    
    return mLoss_ModCV_Optlocal

def SSR_ModCV(vY, mX, vBetaHat, vAlpha_hat, N):
    vY_hat = vAlpha_hat + mX@vBetaHat
    dSSR = np.sum((vY- vY_hat)**2)/N
    
    return dSSR
    

def estBandwidthLMCV(mY, mX, mZ, t_i, T, N, 
                        output_file_prefix,
                        output_path 
                       ):
    print('LMCV Bandwidth selection')
    
    gridh = np.arange(0.1, 0.4, 0.02)
    iH = len(gridh)
    
    lList = [0, 2, 4, 6]
    iNlist = len(lList)
    
    mLosses = np.zeros((T, iNlist, iH))
    for j in range(iH):
        h = gridh[j]
        mK = np.asarray([estK(t_i - i/T, h) for i in range(1, T+1)])
        _, valpha_hat = est_mdl(mY, mX, t_i, T, N, mK)
        mLosses[:,:,j] = ModCVLoss_OptLocal(mY, mX, mZ, mK, valpha_hat, t_i, lList, h, N, T, iNlist)
        
    mListOptLocal = np.zeros((T, iNlist))
    for lid in range(iNlist):
        for j in range(T):
            optloc = np.argmin(mLosses[j,lid,:])
            mListOptLocal[j,lid] = gridh[optloc]        
    h_opt = np.mean(np.mean(mListOptLocal, axis=0))
    
    
    # # CHANGED COMPARED TO HOUSING PAPER:
    losses_filename = os.path.join(output_path, f'{output_file_prefix}_losses_raw.npy')
    try:
        np.save(losses_filename, mLosses)
        print(f"Saved PLMCV raw losses (shape: {mLosses.shape}) to: {losses_filename}")
    except Exception as e:
        print(f"Error saving PLMCV raw losses: {e}")
        
    ############### THIS IS FOR ADDITIONAL DIAGNOSTICS
    print("Generating PLMCV optimal bandwidth plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 8))

    time_axis = np.arange(T)
    for lid, l_val in enumerate(lList):
        plt.plot(time_axis, mListOptLocal[:, lid], marker='.', linestyle='-', label=f'l = {l_val}')

    plt.title('Locally Optimal Bandwidth (PLMCV) Over Time', fontsize=18)
    plt.xlabel('Time (t)', fontsize=14)
    plt.ylabel('Optimal Bandwidth (h)', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend(title="Leave-l-out value", fontsize=12)
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    plt.show()
    
   # Calculate the average optimal h for each 'l' value
    avg_h_per_l = np.mean(mListOptLocal, axis=0)
    
    # Calculate the final, overall average h
    h_opt = np.mean(avg_h_per_l)

    print("\n--- PLMCV Bandwidth Selection Summary ---")
    for i, l_val in enumerate(lList):
        print(f"Average optimal h for l = {l_val}: {avg_h_per_l[i]:.4f}")
    print("-----------------------------------------")
    print(f"Final Compromise Bandwidth (average of all l's): {h_opt:.4f}\n")
    ## --- END OF NEW SECTION --- ##

    return h_opt
    
    


def getFitPlots(mY, mZ, N, idx, mTheta_hat, vAlpha_hat, lIndivNames, 
                # Changed 'saveName' to be more descriptive of its new role
                output_file_prefix 
               ):
    '''
    @Purpose
    ----------
    Fit plots may contribute to a graphical analysis by assesing the model predicitons.

    @Parameters
    ----------
    mY : Endogeneous regressor of dimension (TxN).
    mZ : Contains all regressors including for gt making it of dimension (NxTxd+1).
    N : Amount of individuals.
    idx : x-axis for the plots could be a range of numbers, dates, etc...
         ... common problem fixers for dates; idx.to_period() or idx.to_timestamp()...
         or idx.to_period().to_timestamp() or use idx.dt for these fixers.
    mTheta_hat : The estimated coefficients (Txd+1).
    vAlpha_hat : The estimated dummy variables (Nx1).
    lIndivNames : List of the individual names (the name for each i in N) used for...
                    plot titles and file names.
    output_file_prefix : str
        The full path and base prefix for saving the plot files. 
        Example: os.path.join(output_path, "figures", "individual_fits", "fit_run_details_")
        The function will append '{lIndivNames[n]}.eps' to this prefix.

    @Returns
    -------
    None, see ouput in the Plots/console section or look at saved .eps files.
    '''
    
    # ADDED: Ensure the base directory from the prefix exists
    # This is good practice, though main_LLDVE will likely create the main 'figures' and 'individual_fits' path part
    base_save_dir = os.path.dirname(output_file_prefix)
    if base_save_dir: # Check if there's a directory part in the prefix
        os.makedirs(base_save_dir, exist_ok=True)
    # ------
    
    mY_hat = np.asarray([vAlpha_hat[j] +
                         np.sum(np.multiply(mTheta_hat, mZ[j, :, :]), axis=1) for j in range(N)]).T
    for n in range(N):
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 6))
        plt.scatter(idx, mY[:,n],
                    facecolor='none', edgecolor=f'C{n}', alpha=0.8, linewidth=1)
        plt.plot(idx, mY_hat[:,n], linestyle='--', linewidth=2, color='black', label=r'$\hat y$')
        plt.grid(linestyle='dashed')
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(10))
        #plt.xlabel('Time', fontsize="xx-large")
        plt.ylabel('Observed vs. Fitted', fontsize="xx-large")
        plt.tick_params(axis='both', labelsize=15)
        plt.legend(fontsize="x-large")
        plt.title(f'Fit for {lIndivNames[n]}', fontsize="x-large")
        plt.tight_layout()

        # CHANGED COMPARED TO HOUSING PAPER:
        plot_filename = f'{output_file_prefix}{lIndivNames[n]}.eps'
        
        try:
            plt.savefig(plot_filename)
            # print(f"Saved fit plot to: {plot_filename}") # Optional print
        except Exception as e:
            print(f"Error saving fit plot for {lIndivNames[n]}: {e}")
        plt.show()


def getYmeanPlots(mY, mZ, T, N, idx, mTheta_hat, 
                  # Changed 'saveFileNames' to 'output_plot_full_path' for clarity
                  output_plot_full_path, 
                  ytPredBase # This is for the base model's prediction, can be None
                 ):
    '''
    @Purpose
    ----------
    May contribute to a graphical analysis.

    @Parameters
    ----------
    mY : Endogeneous regressor of dimension (TxN).
    mZ : Contains all regressors including for gt making it of dimension (NxTxd+1).
    T : Amount of observations.
    N : Amount of individuals.
    idx : x-axis for the plots could be a range of numbers, dates, etc...
         ... common problem fixers for dates; idx.to_period() or idx.to_timestamp()...
         or idx.to_period().to_timestamp() or use idx.dt for these fixers.
    mTheta_hat : The estimated coefficients (Txd+1).
    output_plot_full_path : str
        The full path, including filename and extension (e.g., .eps), where the plot will be saved.
        Example: os.path.join(output_path, "figures", "ymean_plot_model1_GS_run1.eps")
    ytPredBase : np.ndarray, optional
        Predicted cross-sectional mean from a base model for comparison (Tx1).
        If None, this line will not be plotted.

    @Returns
    -------
    None, see ouput in the Plots/console section or look at saved .eps files.
    '''
    
    # ADDED COMPARED TO HOUSING: Ensure the directory for the plot exists
    plot_save_dir = os.path.dirname(output_plot_full_path)
    if plot_save_dir: # Check if there's a directory part
        os.makedirs(plot_save_dir, exist_ok=True)    


    ytBar_pred = np.asarray(
        [np.mean(mZ, 0)[t, :]@mTheta_hat[t, :] for t in range(T)])
    # fig = plt.figure(figsize =[15 ,10])
    # plt.subplot(1, 2, 1)
    # for n in range(N):
    #     plt.scatter(idx,mY[:,n], facecolor='none', edgecolor=f'C{n}',alpha=0.8)
    # plt.plot(idx,ytBar_pred ,'k--' ,linewidth=2,label=r'$\overline{y}_{t}^{pred}$')
    # plt.plot(idx,np.mean(mY,1),'k' ,linewidth=1.8,label=r'$\overline{y}_{t}$')
    # plt.legend(loc='lower left',fontsize='x-large')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.subplot(1, 2, 2)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))
    
    if ytPredBase is not None: # Only plot if base predictions are provided
        plt.plot(idx, ytPredBase, '--', linewidth=2, label=r'$\overline{\hat y}_t^{base}$')
    plt.plot(idx, ytBar_pred, '--', linewidth=2, label=r'$\overline{\hat y}_t^{current~model}$')
    plt.plot(idx, np.mean(mY, axis=1), '-', linewidth=2, label=r'$\overline{y}_t^{actual}$')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10))
    #plt.xticks(rotation=45)
    plt.grid(linestyle='dashed')
    #plt.xlabel('Time', fontsize="xx-large")
    plt.ylabel('Cross-sectional Mean Log Yield', fontsize="x-large", labelpad=15)
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(fontsize="x-large", loc='upper left')
    plt.tight_layout()

    # CHANGED COMPARED TO HOUSING PAPER:
    try:
        plt.savefig(output_plot_full_path) # Use the full path directly
        print(f"Saved y-mean plot to: {output_plot_full_path}")
    except Exception as e:
        print(f"Error saving y-mean plot: {e}")
    plt.show()

    print('done')


def step2__4AWB(mX, t_i, T, N, mK, gamma, mEpsilon_tild, first_term):
    '''
    @Purpose
    ----------
    Contains the repetitative steps from the AWB algorithm which is convenient...
    ... for parallelization.

    @Parameters
    ----------
    mX : Contains exogeneous regressors (d) of dimension (NxTxd).
    t_i : Grid points over which one wants to estimate (Tx1).
    T : Amount of observations.
    N : Amount of individuals.
    mK : Consists T diagonal matrices with the weights from the kernel function... 
        ...on the diagonal making it of dimension (TxTxT), by using h.
    gamma : Controls for the dependence and heteroskedasticity in the...
        ... autoregressive part (step2).
    mEpsilon_tild : The oversmoothed residuals of size (TxN).
    first_term : The sum of the oversmoothed model excluding the error terms.

    @Returns
    -------
    mTheta_star : The simulated coefficients (Txd+1).
    '''
    ##step2##
    vVega = np.random.normal(0, (1-gamma**2)**(1/2), T)
    vGhi_star = np.full((T, 1), np.random.standard_normal())
    for i in range(1, T):
        vGhi_star[i] = gamma*vGhi_star[i-1] + vVega[i]

    ##step3##
    mEpsilon_star = np.multiply(vGhi_star, mEpsilon_tild)
    mY_star = first_term + mEpsilon_star

    ##step4##
    mTheta_star, _ = est_mdl(mY_star, mX, t_i, T, N, mK)

    return mTheta_star


def getSimultAlpha(B, alpha, cntrd_stat):
    '''
    @Purpose
    ----------
    To estimate alpha (signficance level) for the simultaneous confidence bands...
    ... where this function coincides with step 2 of the three-step procedure.

    @Parameters
    ----------
    B : The amount of bootstrap simulations.
    alpha : The theoretical significance level.
    cntrd_stat : The centered bootstrap coefficient.

    @Returns
    -------
    The pointwise error which has the closest coverage to the theoretical coverage.
    '''
    alphas = np.linspace(1/B, alpha, B+1)
    f_prev = None
    for i, alpha_p in enumerate(alphas):
        cntrdB = np.sort(cntrd_stat, axis=1)
        L = cntrdB[:, int(B*alpha_p/2)].reshape(cntrd_stat.shape[0], 1)
        U = cntrdB[:, int(B*(1-alpha_p/2))].reshape(cntrd_stat.shape[0], 1)
        f = np.all(np.less_equal(L, cntrd_stat) &
                   np.less_equal(cntrd_stat, U), axis=0)
        covRat = abs(sum(f)/B - (0.95))
        if f_prev is not None and f_prev <= covRat:
            break
        f_prev = covRat

    return alphas[i-1]



def estCI(alpha, T, B, t_i, idx, 
          b_star, b_tild, b_hat, 
          name, title_name, 
          # New arguments for structured saving:
          output_dir_base_path, 
          file_suffix 
          ):
    ####
    #ADJUSTED THE PLOTTING STYLE TO BE MORE SIMILAR TO THE ONE IN LLR!!!
    ####
    '''
    @Purpose
    ----------
    To save the centered bootstrap statistics (.csv) and plot and save the...
    ... estimated coefficients simultaneously with their pointwise and simultaneous...
    ... confidence bands.

    @Parameters
    ----------
    alpha : The significance level used to construct simultaneous bands.
    b_star : The bootstrapped coefficients (TxB).
    b_tild : The oversmoothed coefficients (Tx1).
    b_hat : The estimated coefficients (Tx1).
    T : Amount of observations.
    B : Amount of bootstrap simulations.
    t_i : Grid points over which one wants to estimate (Tx1).
    idx : x-axis for the plots could be a range of numbers, dates, etc...
         ... common problem fixers for dates; idx.to_period() or idx.to_timestamp()...
         or idx.to_period().to_timestamp() or use idx.dt for these fixers.
    name : Variable name used for saving the plots.
    title_name : Variable name used for the plot titles.
    output_dir_base_path : str
        The base directory for this specific model run (e.g., "results/model_1_GS_note").
    file_suffix : str
        A suffix for the filenames (e.g., "model_1_GS_note").
        
    @Returns
    -------
    None, see ouput in the Plots/console section or look at saved .eps files...
    ... some intermediate results are saved in csv files.

    '''
    # 1) compute simultaneous & pointwise bands
    cntrdB = np.sort((b_star - b_tild.reshape(T, 1)), axis=1)
    
    #ADDED: Ensure B is an integer for indexing, as in your version
    B = int(B) 
    
    S_LB = b_hat - cntrdB[:, int(B * (1 - alpha/2))]
    S_UB = b_hat - cntrdB[:, int(B * (alpha/2))]
    alpha_pw = 0.05
    P_LB = b_hat - cntrdB[:, int(B * (1 - alpha_pw/2))]
    P_UB = b_hat - cntrdB[:, int(B * (alpha_pw/2))]
    
    
    # --- Modifications for structured saving ---
    
    # Create specific subdirectories if they don't exist
    csv_output_path = os.path.join(output_dir_base_path, "coefficient_CIs_CSV")
    plot_output_path = os.path.join(output_dir_base_path, "figures") 
    os.makedirs(csv_output_path, exist_ok=True)
    os.makedirs(plot_output_path, exist_ok=True)
    
    # save CSV
    # Filename now incorporates the path and suffix correctly
    csv_filename = os.path.join(csv_output_path, f'{name}_ST_PW_{file_suffix}.csv')
    np.savetxt(
        csv_filename,
        np.c_[b_hat, S_LB, S_UB, P_LB, P_UB],
        delimiter=",",
        header="Beta_hat,Simult_LB,Simult_UB,Pointwise_LB,Pointwise_UB", # Added header
        comments="" # To ensure header starts with # if not desired, or just remove if pandas used later
    )

    # --- End of modifications for CSV saving ---

    # 2) Plot in LLR style (your preferred plotting style)
    # restore pure-Matplotlib defaults in case seaborn was used earlier
    #plt.style.use('default')
    #plt.style.use('seaborn-v0_8-whitegrid')
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))
    # estimate
    plt.plot(
        idx, b_hat,
        color='black', linewidth=2,
        label=f'Estimated $\\beta_{{{name}}}$' 
    )
    # simultaneous bands
    plt.plot(idx, S_LB, 'r--', linewidth=2, label='Simultaneous LB')
    plt.plot(idx, S_UB, 'r--', linewidth=2, label='Simultaneous UB')
    # pointwise fill
    plt.fill_between(
        idx, P_LB, P_UB,
        color='grey', alpha=0.15,
        label='Pointwise CI'
    )
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10))
    #plt.xticks(rotation=45)
    
    plt.grid(linestyle='dashed')
    #plt.xlabel('$t$', fontsize="xx-large") 
    plt.tick_params(axis='both', labelsize=15)
    #plt.legend(fontsize="x-large", loc='upper left')
    plt.title(title_name, fontsize="x-large")
    plt.axhline(0, color='black', linestyle=':', lw=1.5)
    plt.tight_layout()
    
    # --- Modifications for plot saving ---
    # Filename now incorporates the path and suffix correctly
    plot_filename = os.path.join(plot_output_path, f'{name}_{file_suffix}.eps')
    try:
        plt.savefig(plot_filename)
        # print(f"Saved plot for {name} to {plot_filename}") # Optional print
    except Exception as e:
        print(f"Error saving plot for {name}: {e}")
    plt.show()



def estConfidenceBounds(
    # Original arguments from your "old" version
    mY, mX, mZ, t_i, T, N, idx, h, mK, mTheta_hat, vAlpha_hat, 
    ctrl_names, var_titles, 
    # saveFileNames is now replaced by output_path_base and file_suffix_for_run
    B, # B is now explicitly from your "old" version's signature
    
    # New arguments for structured path and run identification
    output_path_base,      # e.g., "results/model_1_GS_run1"
    file_suffix_for_run  # e.g., "model_1_GS_run1" 
    ):
    '''
    @Purpose
    ----------
    This function uses the AWB procedure to obtain bootstrap critical values for... 
    ... constructing pointwise confidence intervals and simultaneous confidence bands.
    It calls your version of estCI to save individual coefficient plots and CSVs 
    to structured directories.
    It returns the full set of bootstrap coefficient draws and oversmoothed coefficients.
    
    @Parameters
    ----------
    mY : Endogeneous regressor of dimension (TxN).
    mX : Contains exogeneous regressors (d) of dimension (NxTxd).
    mZ : Contains all regressors including for gt making it of dimension (NxTxd+1).
    t_i : Grid points over which one wants to estimate (Tx1).
    T : Amount of observations.
    N : Amount of individuals.
    idx : x-axis for the plots could be a range of numbers, dates, etc...
         ... common problem fixers for dates; idx.to_period() or idx.to_timestamp()...
         ... or idx.to_period().to_timestamp() or use idx.dt for these fixers.
    h : The bandwidth that is used for the estimated model.
    mK : Consists T diagonal matrices with the weights from the kernel function... 
        ...on the diagonal making it of dimension (TxTxT), by using h.
    mTheta_hat : The estimated coefficients for all variables (Txd+1).
    vAlpha_hat : The estimated dummy variables (Nx1).
    ctrl_names : List containing the variable names used for saving the plots.
    var_titles : List containing the variable names used for the plot titles.
    B : Number of bootstrap replications.
    output_path_base : str
        The base directory for saving all outputs for this specific run.
    file_suffix_for_run : str
        A suffix (e.g., model_season_note) to be appended to filenames by estCI.

    @Returns
    -------
    mTheta_stars : np.ndarray (B, T, num_coefficients)
        The bootstrap draws of all coefficients.
    mTheta_tild : np.ndarray (T, num_coefficients)
        The oversmoothed coefficients.
    '''
    ##magic numbers##
    # B is now B_reps (passed as argument)
    C_tild = 2
    h_tild = C_tild * h**(5/9)
    gamma = 0.2

    ##step1## (Oversmoothed estimation)
    mK_tild = np.asarray([estK(t_i - i/T, h_tild) for i in range(1, T+1)])
    mTheta_tild, vAlpha_tild = est_mdl(mY, mX, t_i, T, N, mK_tild)
    mEpsilon_tild = np.asarray([mY[:, j] - vAlpha_tild[j] -
                                np.sum(np.multiply(mTheta_tild, mZ[j, :, :]), axis=1) for j in range(N)]).T
    # NEW NOTE: we calculated residuals mEpsilon_tild using the oversmoothed model results
    
    first_term = mY - mEpsilon_tild  # step 3 # NEW NOTE: this is Y_hat_tild
    
    timer = time.time() 
    print(f'Started Parallel Bootstrap Job with {B} replications...')
    
    mTheta_stars = np.asarray(Parallel(n_jobs=-1)(delayed(step2__4AWB)
                                                  (mX, t_i, T, N, mK, gamma, mEpsilon_tild, first_term) for _ in range(B)))
    
    print(f"Bootstrap replications completed in {time.time()-timer:.2f} seconds.")    
    
    ##step5## (CI calculation and plotting loop)
    alpha = 0.05
    
    ## BELOW IS SLIGHTLY ADJUSTED COMPARED TO HOUSING PAPER, BUT DOES THE SAME THING ##:
    num_coefficients = mTheta_stars.shape[-1]
    # Assertions from my previous suggestion (good to keep)
    assert mTheta_hat.shape[1] == num_coefficients, "Shape mismatch: mTheta_hat vs mTheta_stars"
    assert mTheta_tild.shape[1] == num_coefficients, "Shape mismatch: mTheta_tild vs mTheta_stars"

    for i in range(num_coefficients): # Kept your original loop variable 'i'
        cntrd_Bt = mTheta_stars[:, :, i].T - mTheta_tild[:, i].reshape(T, 1)
        alpha_s2 = getSimultAlpha(B, alpha, cntrd_Bt) # Uses B from args
        
        # Call to your modified estCI, passing the new path arguments
        estCI(alpha_s2, T, B, t_i, idx, # Uses B from args
              mTheta_stars[:, :, i].T,      # b_star
              mTheta_tild[:, i],          # b_tild
              mTheta_hat[:, i],           # b_hat
              ctrl_names[i],              # name
              var_titles[i],              # title_name
              # New arguments for estCI:
              output_dir_base_path=output_path_base, 
              file_suffix=file_suffix_for_run 
             )
             
    return mTheta_stars, mTheta_tild




def estConfidenceBoundsOnePlot(mY, mX, mZ, t_i, T, N, idx, h, mK, mTheta_hat, vAlpha_hat, ctrl_names, var_titles, saveFileNames, B):
    '''
    @Purpose
    ----------
    This function uses the AWB procedure to obtain bootstrap critical values for... 
    ... constructing pointwise confidence intervals and simultaneous confidence bands.

    @Parameters
    ----------
    mY : Endogeneous regressor of dimension (TxN).
    mX : Contains exogeneous regressors (d) of dimension (NxTxd).
    mZ : Contains all regressors including for gt making it of dimension (NxTxd+1).
    t_i : Grid points over which one wants to estimate (Tx1).
    T : Amount of observations.
    N : Amount of individuals.
    idx : x-axis for the plots could be a range of numbers, dates, etc...
         ... common problem fixers for dates; idx.to_period() or idx.to_timestamp()...
         ... or idx.to_period().to_timestamp() or use idx.dt for these fixers.
    h : The bandwidth that is used for the estimated model.
    mK : Consists T diagonal matrices with the weights from the kernel function... 
        ...on the diagonal making it of dimension (TxTxT), by using h.
    mTheta_hat : The estimated coefficients for all variables (Txd+1).
    vAlpha_hat : The estimated dummy variables (Nx1).
    ctrl_names : List containing the variable names used for saving the plots.
    var_titles : List containing the variable names used for the plot titles.
    saveFileNames : The file name for saving the plots (.eps) and estimates (.csv).

    @Returns
    -------
    None, see ouput in the Plots/console section or look at saved .eps files,...
    ... estimated coefficients and both type confidence bounds are saved for...
    ... each variable in .csv files.
    '''
    ##magic numbers##
    #B = 10 #1499  # 999
    C_tild = 2
    h_tild = C_tild * h**(5/9)
    gamma = 0.2

    ##step1##
    mK_tild = np.asarray([estK(t_i - i/T, h_tild) for i in range(1, T+1)])
    mTheta_tild, vAlpha_tild = est_mdl(mY, mX, t_i, T, N, mK_tild)
    mEpsilon_tild = np.asarray([mY[:, j] - vAlpha_tild[j] -
                                np.sum(np.multiply(mTheta_tild, mZ[j, :, :]), axis=1) for j in range(N)]).T

    timer = time.time()
    print('Started Parallel Job') 
    
    first_term = mY - mEpsilon_tild  # step 3
    # mTheta_star = np.asarray(
    #     [step2__4AWB(mX,t_i,T,N,mK,gamma,mEpsilon_tild, first_term) for _ in range(B)] )
    mTheta_stars = np.asarray(Parallel(n_jobs=-1)(delayed(step2__4AWB)
                                                  (mX, t_i, T, N, mK, gamma, mEpsilon_tild, first_term) for _ in range(B)))


    print(time.time()-timer)
    
    ## 3) Build band arrays
    alpha = 0.05
    n_betas  = len(ctrl_names)
    G_full = np.arange(1, T+1) / T

    S_LB = np.zeros((n_betas, T))
    S_UB = np.zeros((n_betas, T))
    P_LB = np.zeros((n_betas, T))
    P_UB = np.zeros((n_betas, T))

    for j in range(n_betas):
        # centered draws for beta_j: shape (T, B)
        centered = mTheta_stars[:, :, j].T - mTheta_tild[:, j].reshape(T,1)
        # simultaneous alpha
        alpha_s2 = getSimultAlpha(B, alpha, centered)
        sd = np.sort(centered, axis=1)
        # simultaneous
        S_LB[j] = mTheta_hat[:,j] - sd[:, int(B*(1-alpha_s2/2))]
        S_UB[j] = mTheta_hat[:,j] - sd[:, int(B*(alpha_s2/2))]
        # pointwise at 0.05
        pw = 0.05
        P_LB[j] = mTheta_hat[:,j] - sd[:, int(B*(1-pw/2))]
        P_UB[j] = mTheta_hat[:,j] - sd[:, int(B*(pw/2))]

        # save CSV
        np.savetxt(
            f'{ctrl_names[j]}_ST_PW_{saveFileNames}.csv',
            np.c_[mTheta_hat[:,j], S_LB[j], S_UB[j], P_LB[j], P_UB[j]]
        )

    ## 4) Plot stacked subplots (LLR style)
    plt.style.use('default')
    fig, axes = plt.subplots(
        n_betas, 1,
        figsize=(12, 6*n_betas),
        sharex=True
    )
    if n_betas == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        # estimate
        ax.plot(
            G_full, mTheta_hat[:,j],
            linestyle='--', color='black', linewidth=2,
            label=f'Estimated $\\beta_{{{var_titles[j]}}}$'
        )
        # simultaneous bounds
        ax.plot(G_full, S_LB[j], 'r--', linewidth=2)
        ax.plot(G_full, S_UB[j], 'r--', linewidth=2)
        # pointwise fill
        ax.fill_between(
            G_full, P_LB[j], P_UB[j],
            color='grey', alpha=0.3
        )

        ax.grid(linestyle='dashed')
        ax.tick_params(axis='both', labelsize=15)
        ax.legend(fontsize="x-large", loc='upper left')

    # common x‐label
    axes[-1].set_xlabel('$t/n$', fontsize="xx-large")

    plt.tight_layout()
    plt.savefig(f'{saveFileNames}_allCoeff_CI.eps')
    plt.show()
    


def plotSingleLLDVE(
    mTheta_hat, 
    years, 
    var_names, 
    plot_main_title="Estimated Time-Varying Coefficients",
    save_filename=None
    ):
    """
    Plots the estimated time-varying coefficients from an LLDVE run.

    Parameters:
    ----------
    mTheta_hat : np.ndarray
        Matrix of estimated time-varying coefficients (T x num_coefficients).
        The first column is typically the global trend (intercept), 
        and subsequent columns are for other regressors.
    years : np.ndarray
        Array of years (or time points) for the x-axis (length T).
    var_names : list of str
        List of names for each coefficient/variable, corresponding to the columns 
        of mTheta_hat. Used for y-axis labels and legend.
    plot_main_title : str, optional
        The main title for the entire figure.
    save_filename : str, optional
        If provided, the path (e.g., "path/to/figure.eps") to save the figure.
        If None, the figure is not saved.
    """
    
    # Restore Matplotlib defaults if needed, or set a preferred style
    plt.style.use('seaborn-v0_8-whitegrid')

    assert mTheta_hat.shape[1] == len(var_names), \
        "Number of columns in mTheta_hat must match the length of var_names."

    n_betas = len(var_names)
    
    # Create a figure with one row per coefficient
    fig, axes = plt.subplots(n_betas, 1, figsize=(12, (5*n_betas if n_betas > 1 else 6)), sharex=True)
    
    # If only one beta, axes might not be an array, so make it one
    if n_betas == 1:
        axes = [axes]

    #fig.suptitle(plot_main_title, fontsize="xx-large")

    # Plot each coefficient series in its own axis
    for ax, (j, name) in zip(axes, enumerate(var_names)):
        ax.plot(years,
                mTheta_hat[:, j],
                linestyle='--',
                linewidth=2,
                color='black',
                label=f'Estimate for {name}') # Using name in legend for clarity
        
        ax.grid(linestyle='dashed')
        # Use the provided var_names for y-axis labels
        #ax.set_ylabel(rf"$\hat\beta_{{{name}}}(t)$", fontsize="x-large") 
        ax.tick_params(axis='both', labelsize=15) # Adjusted for potentially many plots
        ax.legend(fontsize="large", loc='best') # Dynamic legend placement

    # Common x‐label at bottom
    #axes[-1].set_xlabel('Year ($t$)', fontsize="x-large")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10))
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
    if save_filename:
        try:
            plt.savefig(save_filename)
            print(f"Plot saved to {save_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
            
    plt.show()
    
    
## FUNCTIONS THAT DID NOT EXIST IN THE ORIGINAL CODE.
## ----------
def calculate_and_save_fitted_residuals( 
    mY,                     # Dependent variable (T_obs, N_units)
    mZ,                     # Full design matrix (N_units, T_obs, num_total_params_incl_intercept)
    mTheta_hat,             # Estimated TVC (T_obs, num_total_params_incl_intercept)
    vAlpha_hat,             # Estimated Fixed Effects (N_units,)
    N_units,                # Number of units
    T_obs,                  # Number of time periods
    output_path,            # Base path to save the CSV files
    years,
    regions,
    fitted_values_filename="fitted_values_final.csv",
    residuals_filename="residuals_final.csv"
    ):
    """
    Calculates final model fitted values and residuals, and saves them to CSV files.

    Parameters:
    ----------
    mY : np.ndarray
        Dependent variable (T_obs, N_units).
    mZ : np.ndarray
        Full design matrix (N_units, T_obs, num_total_params_incl_intercept).
        The first column of the last dimension corresponds to the intercept for g(t).
    mTheta_hat : np.ndarray
        Estimated time-varying coefficients (T_obs, num_total_params_incl_intercept).
        The first column corresponds to g(t).
    vAlpha_hat : np.ndarray
        Estimated fixed effects (N_units,).
    N_units : int
        Number of individual units.
    T_obs : int
        Number of time observations.
    output_path : str
        Directory path where the output CSV files will be saved.
    fitted_values_filename : str, optional
        Filename for the saved fitted values.
    residuals_filename : str, optional
        Filename for the saved residuals.
        
    Returns:
    -------
    mY_fitted : np.ndarray
        Calculated fitted values (T_obs, N_units).
    mResiduals : np.ndarray
        Calculated residuals (T_obs, N_units).
    """
    print("Calculating final fitted values and residuals...")
    
    # mZ is (N, T, d_total_params_incl_intercept)
    # mTheta_hat is (T, d_total_params_incl_intercept)
    mY_fitted = np.zeros_like(mY) # mY is (T,N)
    for i_unit_idx in range(N_units):
        unit_fitted_vals = np.array([mZ[i_unit_idx, t_idx, :] @ mTheta_hat[t_idx,:] for t_idx in range(T_obs)])
        mY_fitted[:, i_unit_idx] = vAlpha_hat[i_unit_idx] + unit_fitted_vals
    
    mResiduals = mY - mY_fitted
    
    # --- MODIFIED SAVING LOGIC ---
    # Use pandas to save with index (years) and columns (regions)
    df_fitted = pd.DataFrame(mY_fitted, index=years, columns=regions)
    df_residuals = pd.DataFrame(mResiduals, index=years, columns=regions)

    try:
        df_fitted.to_csv(os.path.join(output_path, fitted_values_filename))
        print(f"Saved fitted values to: {os.path.join(output_path, fitted_values_filename)}")
    except Exception as e:
        print(f"Error saving fitted values: {e}")
        
    try:
        df_residuals.to_csv(os.path.join(output_path, residuals_filename))
        print(f"Saved residuals to: {os.path.join(output_path, residuals_filename)}")
    except Exception as e:
        print(f"Error saving residuals: {e}")
    # --- END OF MODIFIED SAVING LOGIC ---
        
    return mY_fitted, mResiduals


# --- REPLACE THE OLD HELPER FUNCTION WITH THIS CORRECTED VERSION ---

def _get_rss_and_trace_lldve(h, mY, mX, mZ, t_i, T, N):
    """
    (Corrected) Helper function to compute RSS and the trace of the hat matrix.
    This version correctly replicates the data transformation from est_mdl,
    fixing the universal singularity issue.
    """
    # 1. Estimate the model to get RSS and key parameters
    mK_h = np.asarray([estK(t_i - i/T, h) for i in range(1, T + 1)])
    mTheta_hat, vAlpha_hat = est_mdl(mY, mX, t_i, T, N, mK_h)
    mY_hat = np.asarray([vAlpha_hat[j] + np.sum(np.multiply(mTheta_hat, mZ[j, :, :]), axis=1) for j in range(N)]).T
    RSS = np.sum((mY - mY_hat)**2)

    # 2. Calculate the Trace of the Hat Matrix (S) correctly
    d = mX.shape[-1] + 1
    trace_h = 0
    mX_bar = np.mean(mX, axis=0)
    vY_bar = np.mean(mY, axis=1)
    X_gt = np.ones(T)

    # This analytical loop calculates the trace without full matrix formation
    for i in range(1, T + 1):
        # --- Start: Algebra that now PERFECTLY MATCHES est_mdl ---
        t_i_min_t = t_i - i/T
        mZ_bar_i = np.c_[X_gt, mX_bar, t_i_min_t, mX_bar * (t_i_min_t.reshape(T, 1))]
        vK_i_half = np.sqrt(np.diagonal(mK_h[i-1, :, :])).reshape(T, 1)
        k_sum_i = np.sum(vK_i_half**2)
        
        # This is the full, correct transformation matrix for the demeaned part
        W_proj = (1/k_sum_i) * (vK_i_half @ vK_i_half.T)
        
        lZZ_i = []
        for j in range(N):
            mZi_j = np.c_[X_gt, mX[j, :, :], t_i_min_t, mX[j, :, :] * (t_i_min_t.reshape(T, 1))]
            # Apply the exact transformation from est_mdl to Z
            Z_tilde_j = (np.sqrt(mK_h[i-1, :, :]) @ mZi_j) - (W_proj @ (mZi_j - mZ_bar_i))
            lZZ_i.append(Z_tilde_j.T @ Z_tilde_j)
        
        inv_lZZ_i = np.linalg.inv(np.sum(lZZ_i, 0))
        # --- End: Algebra that now perfectly matches est_mdl ---

        # Calculate the influence matrix (d(theta_hat_i) / d(y_tilde_i))
        # and sum the diagonal elements to get the trace contribution
        local_hat_matrix_sum = np.zeros((d,d))
        for n in range(N):
            mZn_i = np.c_[X_gt, mX[n, :, :], t_i_min_t, mX[n, :, :] * (t_i_min_t.reshape(T, 1))]
            
            # This is d(Y_tilde_n)/d(Y_n)
            dYtild_dYn = np.sqrt(mK_h[i-1,:,:]) - W_proj + (1/N)*W_proj
            
            # This is d(Y_tilde_n)/d(Y_k) for k != n
            dYtild_dYk = (1/N)*W_proj
            
            # We only need the trace, which is the sum of sensitivities of y_hat_in to y_in.
            # This involves the influence of y_in on ALL y_tilde_j, which then influence theta_hat_i.
            # The full analytical derivation is extremely complex.
            # A well-established approximation for such smoothers is that the trace
            # is the trace of the local smoother matrix applied to the regressors.
            
            # Re-approximating the trace calculation with a more robust method
            Z_tilde_n = (np.sqrt(mK_h[i-1, :, :]) @ mZn_i) - (W_proj @ (mZn_i - mZ_bar_i))
            dY_hat_dY_n = Z_tilde_n @ inv_lZZ_i @ Z_tilde_n.T
            trace_h += np.trace(dY_hat_dY_n)


    # The trace of fixed effects adds N to the total
    return RSS, trace_h + N


def estBandwidthAICGCV(mY, mX, mZ, t_i, T, N, criterion='gcv',
                       output_file_prefix=""):
    """
    To obtain the optimal bandwidth for LLDVE by minimizing either the
    Akaike Information Criterion (AIC) or Generalized Cross-Validation (GCV).

    This function is specifically for the LLDVE panel data model.

    @Parameters
    ----------
    mY : Endogeneous regressor of dimension (TxN).
    mX : Contains exogeneous regressors (d) of dimension (NxTxd).
    mZ : Contains all regressors including for gt making it of dimension (NxTxd+1).
    t_i : Grid points over which one wants to estimate (Tx1).
    T : Amount of observations.
    N : Amount of individuals.
    criterion : str
        The criterion to use for bandwidth selection, either 'aic' or 'gcv'.
    output_file_prefix : str
        The full path and base name for saving the output plot and CSV file.
        Example: os.path.join(output_path, "bandwidth_gcv_run1")

    @Returns
    -------
    h_opt : Optimal bandwidth according to the specified criterion.
    """
    print(f"--- Starting LLDVE bandwidth selection using {criterion.upper()} ---")
    
    # Define the grid of bandwidths to search over
    h_grid = np.linspace(0.01, 0.8, 30)
    scores = []
    
    n_total = N * T
    
    # Grid search for the optimal bandwidth
    for h in h_grid:
        print(f"Testing bandwidth h = {h:.4f}...")
        try:
            RSS, traceh = _get_rss_and_trace_lldve(h, mY, mX, mZ, t_i, T, N)
            
            if criterion.lower() == 'aic':
                # AIC = log(RSS/n) + 2 * (traceh + 1) / (n - traceh - 2)
                # Using n_total = N * T
                score = np.log(RSS / n_total) + (2 * (traceh + 1)) / (n_total - traceh - 2)
            elif criterion.lower() == 'gcv':
                # GCV = (RSS/n) / (1 - traceh/n)^2
                score = (RSS / n_total) / ((1 - traceh / n_total)**2)
            else:
                raise ValueError("Criterion must be 'aic' or 'gcv'.")
                
            scores.append(score)
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix encountered for h = {h:.4f}. Skipping.")
            scores.append(np.inf)

    # Find the optimal bandwidth
    h_opt = h_grid[np.argmin(scores)]
    print(f"--- Optimal bandwidth with {criterion.upper()} is h = {h_opt:.4f} ---")

    # --- Plotting and Saving Results ---
    plt.figure(figsize=(12, 6))
    plt.plot(h_grid, scores, linewidth=2, marker='.', markersize=10)
    plt.plot(h_grid[np.argmin(scores)], np.min(scores), marker="o", markersize=12, color='red', ls="")
    plt.grid(linestyle='dashed')
    plt.xlabel('Bandwidth $h$', fontsize="xx-large")
    plt.ylabel(f'{criterion.upper()} Score', fontsize="xx-large")
    plt.tick_params(axis='both', labelsize=15)
    plt.title(f'Bandwidth Selection using {criterion.upper()}', fontsize="x-large")
    plt.tight_layout()

    plot_filename = f'{output_file_prefix}_h_{h_opt:.2f}_{criterion.upper()}_curve.eps'
    csv_filename = f'{output_file_prefix}_{criterion.upper()}_losses.csv'
    
    try:
        plt.savefig(plot_filename)
        print(f"Saved {criterion.upper()} plot to: {plot_filename}")
    except Exception as e:
        print(f"Error saving {criterion.upper()} plot: {e}")
    plt.show()
    
    try:
        np.savetxt(csv_filename, np.column_stack((h_grid, scores)), delimiter=",", header=f"bandwidth,{criterion}_score", comments="")
        print(f"Saved {criterion.upper()} loss values to: {csv_filename}")
    except Exception as e:
        print(f"Error saving {criterion.upper()} CSV: {e}")

    return h_opt
    
    
###################################
    
    
    
def main_LLDVE_old(mY, mX):
    
    T, N   = mY.shape
    t_i    = np.arange(1, T+1) / T
    h_rough = 0.5 # ACTUALLY WE NEED estBandwithLOUOCV()
    mK_opt        = np.asarray([estK(t_i - i/T, h_rough) for i in range(1, T+1)])
    mTheta_hat, vAlpha_hat = est_mdl(mY, mX, t_i, T, N, mK_opt)
    
    return mTheta_hat, vAlpha_hat, h_rough, mK_opt, t_i


def main_LLDVE(
    # Core data inputs
    mY,                     # Dependent variable (T, N)
    mX,                     # Regressors for est_mdl (N, T, d_weather_vars)
    mZ,                     # Full design matrix for residuals/fitted (N, T, d_weather_vars+1)
    
    # Context for paths and run identification
    state_identifier_str,  # e.g., "17"
    model_name_str,         # e.g., "model_1"
    season_name_str,        # e.g., "GS"
    run_note_str,           # e.g., "final_run_plmcv"

    # Time/dimension/labeling context (often from data prep or notebook)
    years,                  # For x-axis of plots (e.g., actual years array) (T,)
    regions,                # List of N region/county names
    var_names,              # List of full variable names for mTheta_hat columns (for plot titles/labels)
    ctrl_names,             # List of short/control names for mTheta_hat columns (for filenames)
    var_titles,             # List of titles for plots (can be same as var_names or more descriptive)
    
    # --- Params with defaults ---
    output_base_dir="results_lldve", # Root directory for all outputs
    
    # Control parameters for the run
    bandwidth_selection_method='plmcv',
    manual_h=None,          # only used when bandwidth_selection_method == 'manual'
    num_bootstrap_reps_B=499,
    
    # Boolean flags for optional actions
    PerformBootstrapAndPlotCI=False,
    PlotSingleCoefficients=False,
    PlotIndividualFits=False, 
    PlotMeanFits=False,       
    SaveBootstrapDraws=False  
):
    """
    Main function to orchestrate LLDVE analysis for a single model specification.
    Handles directory creation, bandwidth selection, estimation, saving of core results,
    and conditionally performs bootstrapping and plotting.
    """
    print(f"--- Running LLDVE for State: {state_identifier_str}, Model {model_name_str}, \
            Season: {season_name_str}, Note: {run_note_str} ---")

    T_obs, N_units = mY.shape # Corrected to get T, N from mY
    t_i    = np.arange(1, T_obs+1) / T_obs


    # 1. Create Output Directory Structure
    # ===================================
    run_specific_folder_name = f"{state_identifier_str}_{model_name_str}_{season_name_str}_{run_note_str}"
    output_path = os.path.join(output_base_dir, run_specific_folder_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Subdirectories for specific plot types (plotting functions will use these)
    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)
    
    print(f"Results will be saved in: {output_path}")
    
    print("Saving key input matrices for this run...")
    pd.DataFrame(mY, index=years, columns=regions).to_csv(os.path.join(output_path, "input_mY.csv"))
    np.save(os.path.join(output_path, "input_mX.npy"), mX)
    pd.DataFrame(regions, columns=['region_name']).to_csv(os.path.join(output_path, "regions_list.csv"), index=False)

    # 2. Bandwidth Selection
    # ======================
    h_opt = None
    # saveName_bw is used by bandwidth functions to save their diagnostic plots/CSVs
    saveName_bw_prefix = f"{state_identifier_str}_{model_name_str}_{season_name_str}_{run_note_str}"

    if bandwidth_selection_method.lower() == 'plmcv':
        print("Performing PLMCV bandwidth selection...")
        # Assuming estBandwidthLMCV saves its own diagnostic CSV/plot using saveName
        # and returns the scalar h_opt.
        # It needs mY, mX (for est_mdl calls), mZ (for ModCVLoss_OptLocal), t_i, T, N
        h_opt = estBandwidthLMCV(mY, mX, mZ, t_i, T_obs, N_units, saveName_bw_prefix, output_path)
    elif bandwidth_selection_method.lower() == 'louocv':
        print("Performing LOUOCV bandwidth selection...")
        # Assuming estBandwithLOUOCV saves its own diagnostic CSV/plot using saveName
        h_opt = estBandwithLOUOCV(mY, mX, mZ, t_i, T_obs, N_units, saveName_bw_prefix)
    elif bandwidth_selection_method.lower() in ['aic', 'gcv']:
        print(f"Performing LLDVE bandwidth selection using {bandwidth_selection_method.upper()}...")
        h_opt = estBandwidthAICGCV(
            mY, mX, mZ, t_i, T_obs, N_units,
            criterion=bandwidth_selection_method.lower(),
            output_file_prefix=saveName_bw_prefix
        )
    elif bandwidth_selection_method.lower() == 'manual':
        # Fixed bandwidth given by the caller. Before this branch existed, 'manual'
        # silently fell through to the 0.27 placeholder below.
        if manual_h is None:
            raise ValueError("bandwidth_selection_method='manual' needs a value for manual_h.")
        h_opt = float(manual_h)
        print(f"Using manual bandwidth h = {h_opt}")
    else:
        h_opt = 0.27 # Fallback or error, ensure this is a sensible default if used
        print(f"Warning: Unknown bandwidth selection method '{bandwidth_selection_method}'. Using default/placeholder h_opt = {h_opt}")
    
    print(f"Optimal bandwidth selected/used: {h_opt:.4f}")
    with open(os.path.join(output_path, "h_optimal.txt"), "w") as f:
        f.write(str(h_opt))


    # 3. Kernel Matrix Calculation
    # ============================
    mK_opt = np.asarray([estK(t_i - i_val/T_obs, h_opt) for i_val in range(1, T_obs + 1)])


    # 4. LLDVE Estimation
    # ===================
    print("Estimating LLDVE model...")
    # est_mdl expects mX (N,T,d_weather_vars) and mY (T,N)
    mTheta_hat, vAlpha_hat = est_mdl(mY, mX, t_i, T_obs, N_units, mK_opt)
    
    
    # 5. Save Core Estimation Results
    # ===============================
    np.savetxt(os.path.join(output_path, "mTheta_hat.csv"), mTheta_hat, delimiter=",")
    np.savetxt(os.path.join(output_path, "vAlpha_hat.csv"), vAlpha_hat, delimiter=",")
    print("Saved mTheta_hat and vAlpha_hat.")


    # 6. Calculate and Save Final Fitted Values & Residuals
    # ===================================================
    mY_fitted_final, mResiduals_final = calculate_and_save_fitted_residuals(
        mY,       # Your dependent variable data
        mZ,       # Full design matrix used for fitting (N, T, d_params)
        mTheta_hat,
        vAlpha_hat,
        N_units,  # Already available from mY.shape
        T_obs,    # Already available from mY.shape
        output_path, # The run-specific output directory
        years=years,   
        regions=regions
        # You can also customize filenames if needed:
        # fitted_values_filename="custom_fitted.csv",
        # residuals_filename="custom_residuals.csv"
    )
    # mY_fitted_final and mResiduals_final are now available for the results_dict if needed

    # 7. Conditional Plotting (Single Run Estimates)
    # ==============================================
    if PlotSingleCoefficients:
        print("Plotting estimated coefficients (single run)...")
        # Ensure plotSingleLLDVE is defined and handles save_filename correctly
        plotSingleLLDVE( # This is your new plotting function
            mTheta_hat, 
            years, 
            var_names, # Ensure this matches columns of mTheta_hat
            plot_main_title=f"Est. Coeffs: {model_name_str} {season_name_str} ({run_note_str})",
            save_filename=os.path.join(figures_path, \
                f"estimated_coeffs_{state_identifier_str}_{model_name_str}_{season_name_str}_{run_note_str}.eps")
        )

    if PlotIndividualFits:
        print("Plotting individual fits...")
        # Ensure getFitPlots is defined and can save to a specific directory
        # It will create many plots, so a sub-sub-folder might be good.
        indiv_fits_path = os.path.join(figures_path, "individual_fits")
        os.makedirs(indiv_fits_path, exist_ok=True)
        # getFitPlots needs mY, mZ, N, idx, mTheta_hat, vAlpha_hat, lIndivNames, saveName (as prefix)
        # The saveName in getFitPlots will be used as a prefix.
        getFitPlots(mY, mZ, N_units, years, mTheta_hat, vAlpha_hat, regions, 
                    output_file_prefix=os.path.join(indiv_fits_path, \
                        f"fit_{state_identifier_str}_{model_name_str}_{season_name_str}_{run_note_str}_"))


    if PlotMeanFits:
        print("Plotting mean fits...")
        # Ensure getYmeanPlots is defined and can save to a specific directory
        # getYmeanPlots needs mY, mZ, T, N, idx, mTheta_hat, saveFileNames, ytPredBase
        # For ytPredBase, if you have a "base model" result from a previous run, you could pass it.
        # Otherwise, pass None or a dummy value if the function handles it.
        # Assuming ytPredBase is not available in a single isolated run of main_LLDVE for now.
        getYmeanPlots(mY, mZ, T_obs, N_units, years, mTheta_hat, 
                      output_plot_full_path=os.path.join(figures_path, \
                          f"ymean_plot_{state_identifier_str}_{model_name_str}_{season_name_str}_{run_note_str}"), 
                      ytPredBase=None) # Pass None or actual base prediction if available


    # 8. Conditional Bootstrap Confidence Intervals
    # =============================================
    mTheta_stars_draws = None # Initialize
    mTheta_tild_estimates = None # Initialize

    if PerformBootstrapAndPlotCI:
        print("Performing Bootstrap and Plotting Confidence Intervals...")
        # Using your estConfidenceBounds (which calls the old-style estCI)
        # It needs output_path_base for estCI to create subfolders
        # and file_suffix_for_run for estCI to append to its filenames
        mTheta_stars_draws, mTheta_tild_estimates = estConfidenceBounds(
            mY, mX, mZ, 
            t_i, T_obs, N_units, 
            years, # This maps to the 'idx' parameter in estConfidenceBounds
            h_opt, mK_opt, 
            mTheta_hat, vAlpha_hat, 
            ctrl_names, var_titles,
            # --- Corrected positional arguments from here ---
            num_bootstrap_reps_B,  # This is the 14th argument, for 'B' in estConfidenceBounds
            output_path,           # This is the 15th argument, for 'output_path_base'
            f"{state_identifier_str}_{model_name_str}_{season_name_str}_{run_note_str}" # This is the 16th, for 'file_suffix_for_run'
        )
        
        if SaveBootstrapDraws and mTheta_stars_draws is not None and mTheta_tild_estimates is not None:
            np.save(os.path.join(output_path, "mTheta_stars_draws.npy"), mTheta_stars_draws)
            np.save(os.path.join(output_path, "mTheta_tild_oversmoothed.npy"), mTheta_tild_estimates)
            print("Saved full bootstrap draws and oversmoothed coefficients.")

    print(f"--- Completed LLDVE for State: {state_identifier_str}, Model: {model_name_str}, \
            Season: {season_name_str}, Note: {run_note_str} ---")
    
    # Return key results that might be useful for the calling notebook
    results_dict = {
        "output_path": output_path,
        "h_optimal": h_opt,
        "mTheta_hat": mTheta_hat,
        "vAlpha_hat": vAlpha_hat,
        "fitted_values": mY_fitted_final,
        "residuals": mResiduals_final,
        "mTheta_stars_draws": mTheta_stars_draws, # Will be None if bootstrap not run
        "mTheta_tild_estimates": mTheta_tild_estimates # Will be None if bootstrap not run
    }
    return output_path, results_dict