import matplotlib.pyplot as plt
from matplotlib.pyplot import hist2d
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import h5py
import time
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2

###################################################################################
def get_data(f_name:str):
    f=h5py.File(f_name,'r')
    data=pd.DataFrame()
    Fir_ID=pd.DataFrame()
    if len(f['Subhalo'])!=0:
        ## Targets
        data['DM_Mass']=f['Subhalo']['SubhaloMassType'][:,1]
        data['DM_HalfmassRad']=f['Subhalo']['SubhaloHalfmassRadType'][:,1]
        data['DM_MassInHalfRad']=f['Subhalo']['SubhaloMassInHalfRadType'][:,1]
        # data['DM_MassInRad']=f['Subhalo']['SubhaloMassInRadType'][:,1]

        ## Features
        # data['PHT_0']=f['Subhalo']['SubhaloStellarPhotometrics'][:,0]
        # data['PHT_1']=f['Subhalo']['SubhaloStellarPhotometrics'][:,1]
        # data['PHT_2']=f['Subhalo']['SubhaloStellarPhotometrics'][:,2]
        # data['PHT_3']=f['Subhalo']['SubhaloStellarPhotometrics'][:,3]
        data['PHT_4']=f['Subhalo']['SubhaloStellarPhotometrics'][:,4]
        data['PHT_5']=f['Subhalo']['SubhaloStellarPhotometrics'][:,5]
        # data['PHT_6']=f['Subhalo']['SubhaloStellarPhotometrics'][:,6]
        # data['PHT_7']=f['Subhalo']['SubhaloStellarPhotometrics'][:,7]

        data['HalfmassRad']=f['Subhalo']['SubhaloHalfmassRadType'][:,4]
        data['MassInHalfRad']=f['Subhalo']['SubhaloMassInHalfRadType'][:,4]
        # data['MassInRad']=f['Subhalo']['SubhaloMassInRadType'][:,4]
        data['Mass']=f['Subhalo']['SubhaloMassType'][:,4]

        data['VelDisp']=f['Subhalo']['SubhaloVelDisp'][:]
        data['Vmax']=f['Subhalo']['SubhaloVmax'][:]

        data['SFR']=f['Subhalo']['SubhaloSFR'][:]
        # SubhaloStarMetallicity, SubhaloStarMetallicityHalfRad
        data['Met']=f['Subhalo']['SubhaloStarMetallicity'][:]
        data['Met_eff']=f['Subhalo']['SubhaloStarMetallicityHalfRad'][:]
        data['SubhaloFlag']=f['Subhalo']['SubhaloFlag'][:]
        data['SubhaloGrNr']=f['Subhalo']['SubhaloGrNr'][:]
        # data['SubhaloParent']=f['Subhalo']['SubhaloParent'][:]
        data['DM_Len']=f['Subhalo']['SubhaloLenType'][:,1]
        data['Len']=f['Subhalo']['SubhaloLenType'][:,4]
    if len(f['Group'])!=0:
        Fir_ID['Fir_ID']=f['Group']['GroupFirstSub'][:]
    f.close()
    del f
    return data,Fir_ID

def main_from_hdf_to_csv(hdf_loc:str,num:int,csv_loc=None):
    data,Fir_ID=pd.DataFrame(),pd.DataFrame()
    for i in range(num):
        name=hdf_loc+str(i)+'.hdf5'
        data_i,Fir_ID_i=get_data(name)
        data=pd.concat([data,data_i],ignore_index=True)
        Fir_ID=pd.concat([Fir_ID,Fir_ID_i],ignore_index=True)
    del data_i,Fir_ID_i
    # add FirSub in data
    Fir_ID_array=Fir_ID.query('Fir_ID>=0').Fir_ID.values
    ID_array=data.index.values
    matched_arrary=np.intersect1d(ID_array,Fir_ID_array)
    array=np.zeros(len(ID_array),bool)
    array[matched_arrary]=True
    data['isCentral']=array
    if type(csv_loc)==str:
        data.to_csv(csv_loc,index_label='ID')
    return data


###################################################################################
def select_use_and_rename_col(data:pd.DataFrame,use_col_ls:list,rename_col_ls:list):
    data_new=data[use_col_ls].copy()
    data_new.columns=rename_col_ls
    return data_new

def get_log_data(data:pd.DataFrame,log_col:list,mass_col:list):
    data_new=data.copy()
    data_new[log_col]=np.log10(data_new[log_col].values)
    data_new[mass_col]=data_new[mass_col]+10
    return data_new

def main_initialize_TNG(loc:str,classify_mod='sSFR',h=0.6774,\
    use_col=['ID','isCentral','DM_MassInHalfRad','PHT_4','PHT_5','HalfmassRad','MassInHalfRad','Mass','VelDisp','Vmax','SFR'],\
    rename_col=['ID','isCentral','DM_MassInHalfRad','g','r','R_eff','MassInHalfRad','Mass','VelDisp','Vmax','SFR']):
    data=pd.read_csv(loc)
    data_robust=data.query('DM_HalfmassRad>2*1 and HalfmassRad>2*1 and SubhaloFlag==True and Len>200 and DM_Len>200').copy()
    data_robust=data_robust[use_col].copy()
    data_robust=select_use_and_rename_col(data_robust,use_col,rename_col)
    data_robust['sSFR']=data_robust['SFR'].values/(data_robust['Mass'].values*1e10/h)
    if classify_mod=='sSFR':
        ETG=data_robust.query('sSFR<1e-11').copy()
        LTG=data_robust.query('sSFR>1e-11').copy()
        TNG=data_robust.copy()
    elif classify_mod=='sSFR_&_VelDisp':
        ETG=data_robust.query('sSFR<1e-11 or VelDisp>150').copy()
        LTG=data_robust.query('sSFR>1e-11 and VelDisp<150').copy()
        TNG=data_robust.copy()
    else: raise ValueError    
    del data, data_robust

    log_col=['DM_MassInHalfRad','R_eff','MassInHalfRad','Mass','VelDisp','Vmax']
    mass_col=['DM_MassInHalfRad','MassInHalfRad','Mass']
    ETG=get_log_data(ETG,log_col,mass_col)
    LTG=get_log_data(LTG,log_col,mass_col)
    TNG=get_log_data(TNG,log_col,mass_col)

    print('The current simulation is {} and classify_mod is {}.'.format('TNG100',classify_mod))
    print('All: {:<8}, ETG: {:<8}, LTG: {:<8}'.format(len(TNG),len(ETG),len(LTG)))
    print('number of central galaxies')
    print('All: {:<8}, ETG: {:<8}, LTG: {:<8}'.format(len(TNG.query('isCentral==True')),len(ETG.query('isCentral==True')),len(LTG.query('isCentral==True'))))
    print('number of satellite galaxies')
    print('All: {:<8}, ETG: {:<8}, LTG: {:<8}'.format(len(TNG.query('isCentral==False')),len(ETG.query('isCentral==False')),len(LTG.query('isCentral==False'))))
    print('{:=^80}'.format('='))

    return ETG,LTG,TNG

def main_initialize_mock(mock_loc,proj=0,classify_mod='sSFR',h=0.6774,z_SBL_ls=None,\
    use_col=['ID','proj','DM_MassInHalfRad','g','r','R_eff','Mass','MassInHalfRad','VelDisp','Vmax','SFR']):
    # 可以加一个选择，是否使用全部的列
    mock=pd.read_csv(mock_loc)
    mock.columns=['DM_HalfmassRad', 'DM_Mass', 'DM_MassInHalfRad', 'DM_VelDisp', 'DM_VelDisp_eff','DM_Vmax','DM_Vmax_eff',
           'proj','R', 'Mass', 'Met', 'Age', 'SFR', 'g', 'r', 'color', 'VelDisp','Vmax',
           'R_eff', 'MassInHalfRad', 'Met_eff', 'Age_eff', 'SFR_eff', 'g_eff',
           'r_eff', 'color_eff', 'VelDisp_eff', 'Vmax_eff', 'ID', 'SubhaloGrNr']
    mock=mock[use_col].copy() 
    mock=mock.query('proj=={} and VelDisp!=0'.format(proj)).copy() ## 得要去掉Veldisp==0的galaxy！！！
    
    mock[['R_eff','VelDisp','Vmax']]=np.log10(mock[['R_eff','VelDisp','Vmax']].values)
    mock['sSFR']=mock['SFR'].values/(10**(mock.Mass.values)/h)
    if classify_mod=='sSFR':
        mock_ETG=mock.query('sSFR<1e-11').copy()
        mock_LTG=mock.query('sSFR>1e-11').copy()
        mock_TNG=mock.copy()
    elif classify_mod=='sSFR_&_VelDisp':
        temp=np.log10(150)
        mock_ETG=mock.query('sSFR<1e-11 or VelDisp>{}'.format(temp)).copy()
        mock_LTG=mock.query('sSFR>1e-11 and VelDisp<{}'.format(temp)).copy()
        mock_TNG=mock.copy()
    else: raise ValueError
    print('The current mock is z={} and SBL={} and classify_mod is {}'.format(*z_SBL_ls,classify_mod))
    print('All: {:<8}, ETG: {:<8}, LTG: {:<8}'.format(len(mock_TNG),len(mock_ETG),len(mock_LTG)))
    print('{:=^80}'.format('='))
    return mock_ETG,mock_LTG,mock_TNG

def get_traintest(data:pd.DataFrame,feature:list,target:str,ts=0.2,ran=1):
    data_feature=data[feature].copy().values
    data_target=data[target].copy().values
    x_train, x_test, y_train, y_test=train_test_split(data_feature,data_target,test_size=ts,random_state=ran)

    if type(target)==str:
        target=[target]
    x_train=pd.DataFrame(data=x_train,columns=feature)
    x_test=pd.DataFrame(data=x_test,columns=feature)
    y_train=pd.DataFrame(data=y_train,columns=target)
    y_test=pd.DataFrame(data=y_test,columns=target)
    
    tr=pd.concat([y_train,x_train],axis=1)
    tt=pd.concat([y_test,x_test],axis=1)
    
    return tr,tt

def get_tr_tt_pred_RF(tr:pd.DataFrame,tt:pd.DataFrame,rf:RandomForestRegressor,\
    feature=['g', 'r', 'R_eff', 'MassInHalfRad', 'Mass', 'VelDisp'],\
    target='DM_MassInHalfRad',title=None,output_print=True,only_tt_result=False):

    tr_new,tt_new=tr.copy(),tt.copy()
    rf.fit(tr_new[feature].values,tr_new[target].values)
    tr_new['pred']=rf.predict(tr_new[feature].values)
    tt_new['pred']=rf.predict(tt_new[feature].values)
    result=pd.DataFrame(columns=['R2_tt','MAE_tt','MSE_tt','pearsonr_tt','R2_tr','MAE_tr','MSE_tr','pearsonr_tr'])
    FI_res=pd.DataFrame(columns=feature)
    FI_res.loc[0,:]=np.round(rf.feature_importances_,4)
    
    true_tt,pred_tt=tt_new[target].values,tt_new['pred'].values
    true_tr,pred_tr=tr_new[target].values,tr_new['pred'].values
    R2_tt,MAE_tt,MSE_tt,pearsonr_tt=R2(true_tt,pred_tt),MAE(true_tt,pred_tt),MSE(true_tt,pred_tt),pearsonr(true_tt,pred_tt)[0]
    R2_tr,MAE_tr,MSE_tr,pearsonr_tr=R2(true_tr,pred_tr),MAE(true_tr,pred_tr),MSE(true_tr,pred_tr),pearsonr(true_tr,pred_tr)[0]
    if output_print==True:
        print('{:+^80}'.format(title))
        print(*list(zip(feature,np.round(rf.feature_importances_,3))))
        print('Prediction in test     set: R2: {:<8.4f}, MAE: {:<8.4f}, MSE: {:<8.4f}, pho: {:<8.4f}'.format(R2_tt,MAE_tt,MSE_tt,pearsonr_tt))
        print('Prediction in training set: R2: {:<8.4f}, MAE: {:<8.4f}, MSE: {:<8.4f}, pho: {:<8.4f}'.format(R2_tr,MAE_tr,MSE_tr,pearsonr_tr))
        print()
    result.loc[0,:]=np.round(np.array([R2_tt,MAE_tt,MSE_tt,pearsonr_tt,R2_tr,MAE_tr,MSE_tr,pearsonr_tr]),4)
    if not only_tt_result:
        return tr_new,tt_new,FI_res,result
    else:
        return tr_new,tt_new,FI_res,result[['R2_tt','MAE_tt','MSE_tt','pearsonr_tt']].copy()

def get_compare_plot_advance(ETG_tt:pd.DataFrame,LTG_tt:pd.DataFrame,TNG_tt:pd.DataFrame,\
        ETG_res:pd.DataFrame,LTG_res:pd.DataFrame,TNG_res:pd.DataFrame,\
        target:str,axlim:tuple,sup_title:str,figsize=(15,5),save_loc=False,\
        plot_mod='hist'):
    title_ls=['ETG','LTG','ALL']
    result_ls=[ETG_res,LTG_res,TNG_res]    # 'R2_tt','MAE_tt','MSE_tt','pearsonr_tt'
    len_ls=[len(ETG_tt),len(LTG_tt),len(TNG_tt)]
    legend_title_ls=[]
    for res,length in zip(result_ls,len_ls):
        R2_tt,MAE_tt,MSE_tt,rho_tt=res.R2_tt.values[0],res.MAE_tt.values[0],res.MSE_tt.values[0],res.pearsonr_tt.values[0]
        name='R2: {:>6.3f}\nMAE: {:>6.3f}\nMSE: {:>6.3f}\nrho: {:>6.3f}\nnumofgal: {:.0f}'.format(R2_tt,MAE_tt,MSE_tt,rho_tt,length)
        legend_title_ls.append(name)
    plt.figure(figsize=figsize,dpi=300)
    if plot_mod=='hist':    
        plt.subplot(131)
        hist2d(ETG_tt[target].values,ETG_tt['pred'],bins=200,norm=LogNorm(),density=False)
        plt.subplot(132)
        hist2d(LTG_tt[target].values,LTG_tt['pred'],bins=200,norm=LogNorm(),density=False)
        plt.subplot(133)
        hist2d(TNG_tt[target].values,TNG_tt['pred'],bins=200,norm=LogNorm(),density=False)
    elif plot_mod=='scatter':
        plt.subplot(131)
        plt.plot(ETG_tt[target],ETG_tt['pred'],'.',)
        plt.subplot(132)
        plt.plot(ETG_tt[target],ETG_tt['pred'],'.',)
        plt.subplot(133)
        plt.plot(TNG_tt[target],TNG_tt['pred'],'.')

    plt.subplot(131)
    plt.plot([axlim[0],axlim[1]],[axlim[0],axlim[1]],'r--')
    plt.xlim(axlim)
    plt.ylim(axlim)
    plt.xlabel('$log(M_{true}/(M_\odot/h))$')
    plt.ylabel('$log(M_{pred}/(M_\odot/h))$')
    # plt.text(9,11,legend_title_ls[0])
    plt.legend(title=legend_title_ls[0],loc='upper left')
    plt.title(title_ls[0])

    plt.subplot(132)
    plt.plot([axlim[0],axlim[1]],[axlim[0],axlim[1]],'r--')
    plt.xlim(axlim)
    plt.ylim(axlim)
    plt.xlabel('$log(M_{true}/(M_\odot/h))$')
    plt.ylabel('$log(M_{pred}/(M_\odot/h))$')
    # plt.text(9,11,legend_title_ls[1])
    plt.legend(title=legend_title_ls[1],loc='upper left')
    plt.title(title_ls[1])

    plt.subplot(133)
    plt.plot([axlim[0],axlim[1]],[axlim[0],axlim[1]],'r--')
    plt.xlim(axlim)
    plt.ylim(axlim)
    plt.xlabel('$log(M_{true}/(M_\odot/h))$')
    plt.ylabel('$log(M_{pred}/(M_\odot/h))$')
    # plt.text(9,11,legend_title_ls[2])
    plt.legend(title=legend_title_ls[2],loc='upper left')
    plt.title(title_ls[2])

    plt.suptitle(sup_title)
    plt.tight_layout()
    if type(save_loc)==str:
        plt.savefig(save_loc)
    plt.show()
    return None

def main_get_result(ETG,LTG,ran_data,ran_tree,axlim=(8.5,12.0),sup_tit=None,save_loc=None,feature_ls=['g','r','R_eff','MassInHalfRad','Mass','VelDisp','Vmax'],**kwargs_rf):
    target='DM_MassInHalfRad'
    ETG_tr,ETG_tt=get_traintest(ETG,feature_ls,target,ts=0.2,ran=ran_data)
    LTG_tr,LTG_tt=get_traintest(LTG,feature_ls,target,ts=0.2,ran=ran_data)
    TNG_tr=pd.concat([ETG_tr,LTG_tr],axis=0,ignore_index=True)
    TNG_tt=pd.concat([ETG_tt,LTG_tt],axis=0,ignore_index=True)

    rf=RandomForestRegressor(n_jobs=-1,max_depth=5,random_state=ran_tree,**kwargs_rf)
    ETG_tr,ETG_tt,ETG_FI_res,ETG_res=get_tr_tt_pred_RF(ETG_tr,ETG_tt,rf,feature_ls,target,'Result of ETG:',output_print=True,only_tt_result=False)
    LTG_tr,LTG_tt,LTG_FI_res,LTG_res=get_tr_tt_pred_RF(LTG_tr,LTG_tt,rf,feature_ls,target,'Result of LTG:',output_print=True,only_tt_result=False)
    TNG_tr,TNG_tt,TNG_FI_res,TNG_res=get_tr_tt_pred_RF(TNG_tr,TNG_tt,rf,feature_ls,target,'Result of TNG:',output_print=True,only_tt_result=False)

    
    get_compare_plot_advance(ETG_tt,LTG_tt,TNG_tt,ETG_res,LTG_res,TNG_res,\
        target=target,sup_title=sup_tit,axlim=axlim,save_loc=save_loc)
    print('{:=^80}'.format('='))
    return None

def get_cross_gal(mock,TNG):
    if len(mock)==len(mock.ID.unique()) and \
        len(TNG)==len(TNG.ID.unique()):
        cross_ID=np.intersect1d(mock.ID.values,TNG.ID.values)
        mock_matched=mock.query('ID in '+str(list(cross_ID))).copy()        
        TNG_matched=TNG.query('ID in '+str(list(cross_ID))).copy()
    # 非常重要，下面这步修改index不能省略，否则main_compare_mock_simu的concat那一步会报错
        mock_matched.index=[i for i in range(len(cross_ID))]
        TNG_matched.index=[i for i in range(len(cross_ID))]
        num_isCentral=len(TNG_matched.query('isCentral==True'))
        num_satellite=len(TNG_matched.query('isCentral==False'))
        print('The values of matched/all galaxies is {}/{}'.format(len(cross_ID),len(mock)))
        print('There are {} central galaxies and {} satellite galaxies'\
            .format(num_isCentral,num_satellite))
        print('{:=^80}'.format('='))
        return mock_matched,TNG_matched
    else: raise ValueError

def main_compare_mock_simu(mock_loc,simu_loc,classify_mod,z_SBL_ls,proj=0,save_loc=False,\
    simu_use_col=['ID','isCentral','SubhaloGrNr','DM_HalfmassRad', 'DM_Mass','DM_MassInHalfRad',\
            'PHT_4','PHT_5','HalfmassRad','MassInHalfRad','Mass','VelDisp','Vmax',\
            'SFR','Met','Met_eff'],\
    simu_rename_col=['ID','isCentral','SubhaloGrNr','DM_HalfmassRad', 'DM_Mass','DM_MassInHalfRad',\
            'g','r','R_eff','MassInHalfRad','Mass','VelDisp','Vmax',\
            'SFR','Met','Met_eff'],\
    mock_use_col=['ID','proj','SubhaloGrNr','DM_HalfmassRad', 'DM_Mass','DM_MassInHalfRad',\
            'g','r','R_eff','MassInHalfRad','Mass','VelDisp','Vmax',\
            'SFR','Met','Met_eff']):
    
    ETG,LTG,TNG=main_initialize_TNG(simu_loc,classify_mod=classify_mod,\
                    use_col=simu_use_col,rename_col=simu_rename_col)
    mock_ETG,mock_LTG,mock_TNG=main_initialize_mock(mock_loc,proj=proj,classify_mod=classify_mod,z_SBL_ls=z_SBL_ls,use_col=mock_use_col)
    print('ETG:')
    mock_ETG_matched,ETG_matched=get_cross_gal(mock_ETG,ETG)
    print('LTG:')
    mock_LTG_matched,LTG_matched=get_cross_gal(mock_LTG,LTG)
    print('TNG:')
    mock_TNG_matched,TNG_matched=get_cross_gal(mock_TNG,TNG)
    print()
    if type(save_loc)==str:
        mock_col_ls=list('mock_'+mock_TNG_matched.columns[2:])
        simu_col_ls=list('simu_'+TNG_matched.columns[2:])
        df_col_ls=['ID','isCentral','proj']+mock_col_ls+simu_col_ls
        mock_ETG_matched.columns=['ID','proj']+mock_col_ls
        mock_LTG_matched.columns=['ID','proj']+mock_col_ls
        mock_TNG_matched.columns=['ID','proj']+mock_col_ls
        ETG_matched.columns=['ID','isCentral']+simu_col_ls
        LTG_matched.columns=['ID','isCentral']+simu_col_ls
        TNG_matched.columns=['ID','isCentral']+simu_col_ls
        ETG_df=pd.concat([mock_ETG_matched[['ID','proj']+mock_col_ls],ETG_matched[['isCentral']+simu_col_ls]],axis=1)
        ETG_df[df_col_ls].copy()
        LTG_df=pd.concat([mock_LTG_matched[['ID','proj']+mock_col_ls],LTG_matched[['isCentral']+simu_col_ls]],axis=1)
        LTG_df[df_col_ls].copy()
        TNG_df=pd.concat([mock_TNG_matched[['ID','proj']+mock_col_ls],TNG_matched[['isCentral']+simu_col_ls]],axis=1)
        TNG_df[df_col_ls].copy()
        ETG_df.to_csv(save_loc+'ETG_matched_z{}_SBL{}.csv'.format(*z_SBL_ls),index=False)
        LTG_df.to_csv(save_loc+'LTG_matched_z{}_SBL{}.csv'.format(*z_SBL_ls),index=False)
        TNG_df.to_csv(save_loc+'TNG_matched_z{}_SBL{}.csv'.format(*z_SBL_ls),index=False)
        return ETG_df,LTG_df,TNG_df
    if save_loc==False:
        return mock_ETG_matched,mock_LTG_matched,mock_TNG_matched,ETG_matched,LTG_matched,TNG_matched


################################################################################################
# # 暂时不需要的
# def select_use_and_rename_col(data:pd.DataFrame,use_col_ls:list,rename_col_ls:list):
#     data_new=data[use_col_ls].copy()
#     data_new.columns=rename_col_ls
#     return data_new
# def initialize(loc:str,h=0.6774):
#     data=pd.read_csv(loc)
#     print('The current simulation is {}'.format('TNG100'))
#     # data_robust=data[(data.SubhaloFlag==True)&(data.DM_HalfmassRad>2*1)&(data.HalfmassRad>2*1)&(data.Len>200)&(data.DM_Len>200)].copy()
#     data_robust=data.query('DM_HalfmassRad>2*1 and HalfmassRad>2*1 and SubhaloFlag==True and Len>200 and DM_Len>200').copy()
#     data_robust['sSFR']=data_robust['SFR'].values/(data_robust['Mass'].values*1e10/h)
#     ETG=data_robust.query('sSFR<1e-11').copy()
#     LTG=data_robust.query('sSFR>1e-11').copy()
#     TNG=data_robust.copy()
#     del data, data_robust

#     use_col=['ID','FirSub','DM_MassInHalfRad','PHT_4','PHT_5','HalfmassRad','MassInHalfRad','Mass','VelDisp','Vmax','SFR']
#     rename_col=['ID','isCentral','DM_MassInHalfRad','g','r','R_eff','MassInHalfRad','Mass','VelDisp','Vmax','SFR']
#     ETG=select_use_and_rename_col(ETG,use_col,rename_col)
#     LTG=select_use_and_rename_col(LTG,use_col,rename_col)
#     TNG=select_use_and_rename_col(TNG,use_col,rename_col)

#     log_col=['DM_MassInHalfRad','R_eff','MassInHalfRad','Mass','VelDisp','Vmax']
#     mass_col=['DM_MassInHalfRad','MassInHalfRad','Mass']
#     ETG=get_log_data(ETG,log_col,mass_col)
#     LTG=get_log_data(LTG,log_col,mass_col)
#     TNG=get_log_data(TNG,log_col,mass_col)

#     print('All: {:<8}, ETG: {:<8}, LTG: {:<8}'.format(len(TNG),len(ETG),len(LTG)))
#     print('number of central galaxies')
#     print('All: {:<8}, ETG: {:<8}, LTG: {:<8}'.format(len(TNG.query('isCentral==True')),len(ETG.query('isCentral==True')),len(LTG.query('isCentral==True'))))
#     print('number of satellite galaxies')
#     print('All: {:<8}, ETG: {:<8}, LTG: {:<8}'.format(len(TNG.query('isCentral==False')),len(ETG.query('isCentral==False')),len(LTG.query('isCentral==False'))))
#     print('{:=^80}'.format('='))

#     return ETG,LTG,TNG